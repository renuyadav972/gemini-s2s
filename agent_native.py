"""
Native Gemini 3.1 Flash Live Agent
====================================
Single model handles speech-to-speech natively.
No separate STT or TTS. One API connection per turn.

Uses pipecat's GeminiLiveLLMService with Plivo telephony transport.

Run:
    python agent_native.py -t plivo -x <ngrok-host> --port 8001
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMContextFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from metrics_observer import MetricsCollectorObserver

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Language config
# ---------------------------------------------------------------------------

LANGUAGE_CONFIG = {
    "hi": {
        "name": "Hindi",
        "greeting": "नमस्ते! मैं आपकी पर्सनल असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकती हूँ?",
    },
    "ta": {
        "name": "Tamil",
        "greeting": "வணக்கம்! நான் உங்கள் பர்சனல் அசிஸ்டென்ட். நான் உங்களுக்கு எப்படி உதவ முடியும்?",
    },
    "bn": {
        "name": "Bengali",
        "greeting": "নমস্কার! আমি আপনার পার্সোনাল অ্যাসিস্ট্যান্ট। আমি কিভাবে সাহায্য করতে পারি?",
    },
    "en": {
        "name": "English",
        "greeting": "Hello! I'm your personal assistant. How can I help you?",
    },
}


def get_system_prompt(language: str) -> str:
    lang_name = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["hi"])["name"]
    now = datetime.now(timezone.utc)
    return (
        f"You are a helpful personal assistant on a phone call. "
        f"The current date is {now.strftime('%B %d, %Y')} and time is {now.strftime('%I:%M %p')} UTC.\n\n"
        f"You can help with:\n"
        f"- Greeting and casual conversation\n"
        f"- Math calculations\n"
        f"- Language help and translations\n"
        f"- General knowledge you are confident about\n\n"
        f"Rules:\n"
        f"- If you are not sure about something, say so honestly. Do not make up facts.\n"
        f"- Never guess current weather, news, sports scores, or stock prices.\n"
        f"- Keep responses to 1-2 short sentences. This is a phone call.\n"
        f"- Respond in whatever language the caller speaks. If they switch languages, switch with them.\n"
        f"- No markdown, bullets, or formatting. Your words will be spoken aloud.\n"
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def run_bot(
    transport: BaseTransport,
    handle_sigint: bool,
    language: str = "hi",
    call_id: str | None = None,
):
    lang_config = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["hi"])

    session_id = str(uuid.uuid4())
    data_dir = os.path.join(os.path.dirname(__file__), "data", "sessions")

    metrics_observer = MetricsCollectorObserver(
        session_id=session_id,
        mode="native",
        config={"language": language, "model": "gemini-3.1-flash-live"},
        data_dir=data_dir,
    )

    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        system_instruction=get_system_prompt(language),
        voice_id="Aoede",
        model="models/gemini-3.1-flash-live-preview",
    )

    pipeline = Pipeline(
        [
            transport.input(),
            llm,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
            observers=[metrics_observer],
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        if call_id:
            asyncio.create_task(_start_recording(call_id))
        logger.info(f"Client connected — native mode, language={language}")
        await asyncio.sleep(1.5)
        # Send initial context to trigger setup and enable audio input
        context = LLMContext(
            messages=[{"role": "user", "content": "Greet the caller warmly in " + lang_config["name"]}]
        )
        await task.queue_frames([LLMContextFrame(context=context)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if call_id:
            asyncio.create_task(_fetch_recording(call_id, session_id, data_dir))
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


# ---------------------------------------------------------------------------
# Plivo recording helpers (shared pattern)
# ---------------------------------------------------------------------------


async def _start_recording(call_id: str):
    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        return
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"https://api.plivo.com/v1/Account/{auth_id}/Call/{call_id}/Record/",
                auth=(auth_id, auth_token),
                json={"time_limit": 300, "file_format": "mp3"},
            )
            logger.info(f"Plivo recording started: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to start recording: {e}")


async def _fetch_recording(call_id: str, session_id: str, data_dir: str):
    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        return
    import json

    for attempt in range(12):
        await asyncio.sleep(5)
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"https://api.plivo.com/v1/Account/{auth_id}/Recording/",
                    auth=(auth_id, auth_token),
                    params={"call_uuid": call_id, "limit": 1},
                )
                if resp.status_code == 200:
                    objects = resp.json().get("objects", [])
                    if objects:
                        recording_url = objects[0].get("recording_url")
                        if recording_url:
                            logger.info(f"Recording ready: {recording_url}")
                            dl = await http.get(recording_url)
                            rec_path = os.path.join(data_dir, f"{session_id}.mp3")
                            with open(rec_path, "wb") as f:
                                f.write(dl.content)
                            logger.info(f"Recording saved: {rec_path}")
                            session_path = os.path.join(data_dir, f"{session_id}.json")
                            if os.path.exists(session_path):
                                with open(session_path) as f:
                                    session = json.load(f)
                                session["recording_url"] = recording_url
                                session["recording_file"] = rec_path
                                with open(session_path, "w") as f:
                                    json.dump(session, f, indent=2)
                            return
        except Exception as e:
            logger.warning(f"Recording fetch attempt {attempt + 1}: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def bot(runner_args: RunnerArguments):
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Native agent: transport={transport_type}")

    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_passthrough=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    language = os.getenv("ASR_LANGUAGE", "hi")
    await run_bot(transport, runner_args.handle_sigint, language=language, call_id=call_data["call_id"])


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

"""
Cascaded Pipeline Agent
========================
Parakeet RNNT 1.1B (STT) → Gemini 2.5 Flash (LLM) → ElevenLabs (TTS)

Three separate models, three API calls per turn.
Uses pipecat with Plivo telephony transport.

Run:
    python agent_cascaded.py -t plivo -x <ngrok-host> --port 8000
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
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
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.nvidia.stt import NvidiaSegmentedSTTService, language_to_nvidia_riva_language
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from metrics_observer import MetricsCollectorObserver

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Parakeet STT with extended Indic language support
# ---------------------------------------------------------------------------

# Pipecat's built-in map only has Hindi. Add Tamil, Bengali, Telugu.
INDIC_LANGUAGE_MAP = {
    Language.HI: "hi-IN",
    Language.HI_IN: "hi-IN",
    Language.TA: "ta-IN",
    Language.TA_IN: "ta-IN",
    Language.BN: "bn-IN",
    Language.BN_IN: "bn-IN",
    Language.EN: "en-US",
    Language.EN_US: "en-US",
}


class ParakeetIndicSTTService(NvidiaSegmentedSTTService):
    """Parakeet RNNT 1.1B with Tamil, Bengali, Telugu support."""

    def language_to_service_language(self, language):
        return INDIC_LANGUAGE_MAP.get(language) or language_to_nvidia_riva_language(language)

    def _get_language_code(self) -> str:
        """Return Riva language code string (e.g., 'ta-IN'), not enum."""
        lang = self._settings.language
        if lang:
            mapped = self.language_to_service_language(lang)
            if mapped:
                return mapped
            return str(lang)
        return "en-US"

    def _create_recognition_config(self):
        """Override to force string language code and set encoding."""
        import riva.client

        lang_code = str(self._get_language_code())
        logger.info(f"Parakeet config: language_code={lang_code}, sample_rate={self.sample_rate}")

        config = riva.client.RecognitionConfig(
            language_code=lang_code,
            max_alternatives=1,
            profanity_filter=self._settings.profanity_filter,
            enable_automatic_punctuation=self._settings.automatic_punctuation,
            verbatim_transcripts=self._settings.verbatim_transcripts,
            audio_channel_count=1,
        )
        config.encoding = riva.client.AudioEncoding.LINEAR_PCM
        config.sample_rate_hertz = self.sample_rate or 16000
        return config


# ---------------------------------------------------------------------------
# Language config
# ---------------------------------------------------------------------------

LANGUAGE_CONFIG = {
    "hi": {
        "name": "Hindi",
        "stt_language": Language.HI_IN,
        "greeting": "नमस्ते! मैं आपकी पर्सनल असिस्टेंट हूँ। मैं आपकी कैसे मदद कर सकती हूँ?",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah - natural female
    },
    "ta": {
        "name": "Tamil",
        "stt_language": Language.TA_IN,
        "greeting": "வணக்கம்! நான் உங்கள் பர்சனல் அசிஸ்டென்ட். நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah - multilingual
    },
    "bn": {
        "name": "Bengali",
        "stt_language": Language.BN_IN,
        "greeting": "নমস্কার! আমি আপনার পার্সোনাল অ্যাসিস্ট্যান্ট। আমি কিভাবে সাহায্য করতে পারি?",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah - multilingual
    },
    "en": {
        "name": "English",
        "stt_language": Language.EN_US,
        "greeting": "Hello! I'm your personal assistant. How can I help you?",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah
    },
}

# Parakeet RNNT 1.1B Multilingual (Indic profile)
PARAKEET_INDIC_FUNCTION_MAP = {
    "function_id": "71203149-d3b7-4460-8231-1be2543a1fca",
    "model_name": "parakeet-1.1b-rnnt-multilingual-asr",
}


def get_system_prompt(language: str) -> str:
    lang_name = LANGUAGE_CONFIG.get(language, LANGUAGE_CONFIG["hi"])["name"]
    now = datetime.now(timezone.utc)
    return (
        f"You are a helpful personal assistant on a phone call. "
        f"You speak {lang_name} fluently.\n"
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
        f"- Always respond in {lang_name} unless asked otherwise.\n"
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
        mode="cascaded",
        config={"language": language, "stt": "parakeet-rnnt-1.1b", "llm": "gemini-2.5-flash", "tts": "elevenlabs-multilingual-v2"},
        data_dir=data_dir,
    )

    stt = ParakeetIndicSTTService(
        api_key=os.getenv("NVIDIA_API_KEY", ""),
        model_function_map=PARAKEET_INDIC_FUNCTION_MAP,
        params=ParakeetIndicSTTService.InputParams(
            language=lang_config["stt_language"],
        ),
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash",
    )

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=lang_config["voice_id"],
        model="eleven_flash_v2_5",
    )

    messages = [{"role": "system", "content": get_system_prompt(language)}]
    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.7)],
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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
        logger.info(f"Client connected — language={language}, greeting in {lang_config['name']}")
        await asyncio.sleep(1.5)
        await task.queue_frames([TTSSpeakFrame(text=lang_config["greeting"])])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if call_id:
            asyncio.create_task(_fetch_recording(call_id, session_id, data_dir))
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


# ---------------------------------------------------------------------------
# Plivo recording helpers
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
    logger.info(f"Cascaded agent: transport={transport_type}")

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
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # Language from query params (set on Plivo app answer_url) or env fallback
    ws_path = runner_args.websocket.scope.get("path", "")
    qs = runner_args.websocket.scope.get("query_string", b"").decode()
    language = os.getenv("ASR_LANGUAGE", "hi")
    # Parse ?language=xx from query string if present
    for part in qs.split("&"):
        if part.startswith("language="):
            language = part.split("=", 1)[1]
            break
    logger.info(f"Language from request: {language}")
    await run_bot(transport, runner_args.handle_sigint, language=language, call_id=call_data["call_id"])


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

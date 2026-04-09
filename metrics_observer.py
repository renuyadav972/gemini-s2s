"""
Metrics Collector Observer
==========================
BaseObserver subclass that captures per-turn metrics during a Pipecat voice
agent session and writes them as JSON to disk.

Captures the full latency waterfall per turn:
  user_stopped → [STT TTFB] → [LLM TTFB] → [TTS TTFB] → bot_started

Attach to a PipelineTask via PipelineParams(observers=[...]).
"""

import json
import os
import tempfile
import time
from collections import deque
from datetime import datetime, timezone

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    SmartTurnMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


class MetricsCollectorObserver(BaseObserver):
    """Captures per-turn latency waterfall, transcriptions, and Smart Turn data.

    Writes a session JSON file to ``data_dir/{session_id}.json`` after every
    finalized turn, using atomic rename for safe concurrent reads.
    """

    def __init__(
        self,
        *,
        session_id: str,
        mode: str,
        config: dict,
        data_dir: str = "data/sessions",
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._session_id = session_id
        self._mode = mode
        self._config = config
        self._data_dir = data_dir
        self._output_path = os.path.join(data_dir, f"{session_id}.json")

        # Frame dedup — bounded set backed by a deque
        self._processed_frames: set = set()
        self._frame_history: deque = deque(maxlen=200)

        # Session state
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._ended_at = None
        self._turns: list[dict] = []
        self._connection_time: float | None = None

        # Current turn accumulators
        self._user_started_at: float = 0
        self._user_stopped_at: float = 0
        self._bot_started_at: float = 0
        self._user_text_parts: list[str] = []
        self._bot_text_parts: list[str] = []
        self._smart_turn_data: dict | None = None
        self._turn_number: int = 0

        # Per-turn TTFB accumulators (last value per processor type per turn)
        self._stt_ttfb: float | None = None
        self._llm_ttfb: float | None = None
        self._tts_ttfb: float | None = None
        self._stt_processing: float | None = None
        self._llm_usage: dict | None = None
        self._tts_usage: int | None = None

        os.makedirs(data_dir, exist_ok=True)
        logger.info(
            f"MetricsCollector: session={session_id} mode={mode} -> {self._output_path}"
        )

    # ------------------------------------------------------------------
    # Dedup helper
    # ------------------------------------------------------------------

    def _is_duplicate(self, frame_id) -> bool:
        if frame_id in self._processed_frames:
            return True
        self._processed_frames.add(frame_id)
        self._frame_history.append(frame_id)
        if len(self._processed_frames) > len(self._frame_history) + 50:
            self._processed_frames = set(self._frame_history)
        return False

    # ------------------------------------------------------------------
    # Observer callback
    # ------------------------------------------------------------------

    async def on_push_frame(self, data: FramePushed):
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        if self._is_duplicate(data.frame.id):
            return

        frame = data.frame

        if isinstance(frame, (VADUserStartedSpeakingFrame, UserStartedSpeakingFrame)):
            self._on_user_started(frame)
        elif isinstance(frame, (VADUserStoppedSpeakingFrame, UserStoppedSpeakingFrame)):
            self._on_user_stopped(frame)
        elif isinstance(frame, TranscriptionFrame):
            self._on_transcription(frame)
        elif isinstance(frame, LLMTextFrame):
            self._on_llm_text(frame)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._on_bot_started()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._on_bot_stopped()
        elif isinstance(frame, MetricsFrame):
            self._on_metrics(frame)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._finalize_session()

    # ------------------------------------------------------------------
    # Frame handlers
    # ------------------------------------------------------------------

    def _on_user_started(self, frame):
        # VAD frames have timestamp; Flux frames may not
        ts = getattr(frame, "timestamp", None) or time.time()
        # Record first VAD event as connection time
        if self._connection_time is None:
            self._connection_time = ts

        self._user_started_at = ts
        self._user_stopped_at = 0
        self._bot_started_at = 0
        self._user_text_parts = []
        self._bot_text_parts = []
        self._smart_turn_data = None
        # Reset per-turn TTFB accumulators
        self._stt_ttfb = None
        self._llm_ttfb = None
        self._tts_ttfb = None
        self._stt_processing = None
        self._llm_usage = None
        self._tts_usage = None

    def _on_user_stopped(self, frame):
        # VAD frames have timestamp + stop_secs; Flux frames may not
        ts = getattr(frame, "timestamp", None) or time.time()
        stop_secs = getattr(frame, "stop_secs", 0)
        self._user_stopped_at = ts - stop_secs

    def _on_transcription(self, frame: TranscriptionFrame):
        if frame.text and frame.text.strip():
            self._user_text_parts.append(frame.text.strip())

    def _on_llm_text(self, frame: LLMTextFrame):
        if frame.text:
            self._bot_text_parts.append(frame.text)

    def _on_bot_started(self):
        now = time.time()
        self._bot_started_at = now
        if self._user_stopped_at and self._user_started_at:
            latency_ms = round((now - self._user_stopped_at) * 1000)
        else:
            latency_ms = 0

        self._turn_number += 1
        turn = {
            "turn_number": self._turn_number,
            "user_started_at": self._user_started_at or None,
            "user_stopped_at": self._user_stopped_at or None,
            "bot_started_at": now,
            "bot_stopped_at": None,
            "response_latency_ms": max(latency_ms, 0),
            "user_text": " ".join(self._user_text_parts),
            "bot_text": "",
            "pipeline": {
                "stt_ttfb_ms": round(self._stt_ttfb * 1000, 1) if self._stt_ttfb else None,
                "stt_processing_ms": round(self._stt_processing * 1000, 1) if self._stt_processing else None,
                "llm_ttfb_ms": round(self._llm_ttfb * 1000, 1) if self._llm_ttfb else None,
                "tts_ttfb_ms": round(self._tts_ttfb * 1000, 1) if self._tts_ttfb else None,
            },
        }
        if self._llm_usage:
            turn["pipeline"]["llm_prompt_tokens"] = self._llm_usage.get("prompt_tokens", 0)
            turn["pipeline"]["llm_completion_tokens"] = self._llm_usage.get("completion_tokens", 0)
        if self._tts_usage is not None:
            turn["pipeline"]["tts_characters"] = self._tts_usage
        if self._smart_turn_data:
            turn["smart_turn"] = self._smart_turn_data
        self._turns.append(turn)
        logger.info(
            f"MetricsCollector: turn {self._turn_number} "
            f"latency={latency_ms}ms "
            f"stt={turn['pipeline']['stt_ttfb_ms']}ms "
            f"llm={turn['pipeline']['llm_ttfb_ms']}ms "
            f"tts={turn['pipeline']['tts_ttfb_ms']}ms"
        )

    def _on_bot_stopped(self):
        now = time.time()
        if self._turns:
            self._turns[-1]["bot_text"] = "".join(self._bot_text_parts)
            self._turns[-1]["bot_stopped_at"] = now
        self._bot_text_parts = []
        self._flush()

    def _on_metrics(self, frame: MetricsFrame):
        for m in frame.data:
            if isinstance(m, SmartTurnMetricsData):
                self._smart_turn_data = {
                    "is_complete": m.is_complete,
                    "probability": round(m.probability, 4),
                    "inference_time_ms": round(m.inference_time_ms, 1),
                    "e2e_processing_time_ms": round(m.e2e_processing_time_ms, 1),
                }
            elif isinstance(m, TTFBMetricsData):
                proc = m.processor.lower()
                val = m.value  # seconds
                if "stt" in proc or "deepgram" in proc or "speech" in proc or "nvidia" in proc or "parakeet" in proc:
                    self._stt_ttfb = val
                elif "llm" in proc or "google" in proc or "anthropic" in proc or "openai" in proc or "gemini" in proc:
                    self._llm_ttfb = val
                elif "tts" in proc or "elevenlabs" in proc or "cartesia" in proc:
                    self._tts_ttfb = val
            elif isinstance(m, ProcessingMetricsData):
                proc = m.processor.lower()
                if "stt" in proc or "deepgram" in proc or "speech" in proc or "nvidia" in proc or "parakeet" in proc:
                    self._stt_processing = m.value
            elif isinstance(m, LLMUsageMetricsData):
                self._llm_usage = {
                    "prompt_tokens": m.value.prompt_tokens,
                    "completion_tokens": m.value.completion_tokens,
                }
            elif isinstance(m, TTSUsageMetricsData):
                self._tts_usage = m.value

    def _finalize_session(self):
        self._ended_at = datetime.now(timezone.utc).isoformat()
        self._flush()
        logger.info(
            f"MetricsCollector: session {self._session_id} finalized — "
            f"{len(self._turns)} turns"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict:
        latencies = [t["response_latency_ms"] for t in self._turns if t["response_latency_ms"]]
        smart_probs = [
            t["smart_turn"]["probability"]
            for t in self._turns
            if t.get("smart_turn")
        ]
        smart_infer = [
            t["smart_turn"]["inference_time_ms"]
            for t in self._turns
            if t.get("smart_turn")
        ]

        # Pipeline averages
        stt_vals = [t["pipeline"]["stt_ttfb_ms"] for t in self._turns if t.get("pipeline", {}).get("stt_ttfb_ms")]
        llm_vals = [t["pipeline"]["llm_ttfb_ms"] for t in self._turns if t.get("pipeline", {}).get("llm_ttfb_ms")]
        tts_vals = [t["pipeline"]["tts_ttfb_ms"] for t in self._turns if t.get("pipeline", {}).get("tts_ttfb_ms")]

        summary = {
            "total_turns": len(self._turns),
            "avg_response_latency_ms": round(sum(latencies) / len(latencies)) if latencies else 0,
            "min_response_latency_ms": min(latencies) if latencies else 0,
            "max_response_latency_ms": max(latencies) if latencies else 0,
            "avg_stt_ttfb_ms": round(sum(stt_vals) / len(stt_vals), 1) if stt_vals else None,
            "avg_llm_ttfb_ms": round(sum(llm_vals) / len(llm_vals), 1) if llm_vals else None,
            "avg_tts_ttfb_ms": round(sum(tts_vals) / len(tts_vals), 1) if tts_vals else None,
        }
        if smart_probs:
            summary["avg_smart_turn_probability"] = round(
                sum(smart_probs) / len(smart_probs), 4
            )
        if smart_infer:
            summary["avg_smart_turn_inference_ms"] = round(
                sum(smart_infer) / len(smart_infer), 1
            )

        # ── New quality / impact metrics ──────────────────────────────
        re_prompt_patterns = ("sorry", "repeat", "didn't catch", "didn't get", "say that again", "come again", "pardon")
        interruption_count = 0
        false_endpoint_count = 0
        reprompt_count = 0
        num_turns = len(self._turns)

        for idx, t in enumerate(self._turns):
            # Per-turn flags (defaults)
            t["is_interruption"] = False
            t["is_false_endpoint"] = False
            t["is_reprompt"] = False

            # Incorrect dead air: latency beyond pipeline processing
            lat = t["response_latency_ms"]
            p = t.get("pipeline", {})
            pipeline_ms = (p.get("stt_ttfb_ms") or 0) + (p.get("llm_ttfb_ms") or 0) + (p.get("tts_ttfb_ms") or 0)
            unnecessary = max(lat - pipeline_ms, 0)
            t["pipeline_ms"] = round(pipeline_ms, 1)
            t["unnecessary_dead_air_ms"] = round(unnecessary, 1)
            t["pct_incorrect_dead_air"] = round(unnecessary / lat * 100, 1) if lat > 0 else 0

            # Interruption: user N+1 started before bot N finished speaking
            if idx > 0:
                prev = self._turns[idx - 1]
                prev_bot_stopped = prev.get("bot_stopped_at")
                cur_user_started = t.get("user_started_at")
                if prev_bot_stopped and cur_user_started and cur_user_started < prev_bot_stopped:
                    t["is_interruption"] = True
                    interruption_count += 1

            # False endpoint: short gap between consecutive user utterances (mid-sentence cutoff)
            if idx > 0:
                prev = self._turns[idx - 1]
                prev_user_stopped = prev.get("user_stopped_at")
                cur_user_started = t.get("user_started_at")
                if prev_user_stopped and cur_user_started:
                    gap = cur_user_started - prev_user_stopped
                    if 0 < gap < 2.0:
                        t["is_false_endpoint"] = True
                        false_endpoint_count += 1

            # Re-prompt: bot apologized / asked to repeat
            bot_text_lower = (t.get("bot_text") or "").lower()
            if any(p in bot_text_lower for p in re_prompt_patterns):
                t["is_reprompt"] = True
                reprompt_count += 1

        denom_pairs = max(num_turns - 1, 1)
        summary["interruption_count"] = interruption_count
        summary["interruption_rate"] = round(interruption_count / denom_pairs, 4)
        summary["false_endpoint_count"] = false_endpoint_count
        summary["false_endpoint_rate"] = round(false_endpoint_count / denom_pairs, 4)
        summary["reprompt_count"] = reprompt_count
        summary["reprompt_rate"] = round(reprompt_count / max(num_turns, 1), 4)

        # Dead air: total response latency across all turns
        dead_air_ms = sum(t["response_latency_ms"] for t in self._turns if t["response_latency_ms"])
        summary["dead_air_ms"] = dead_air_ms
        summary["dead_air_s"] = round(dead_air_ms / 1000, 2)

        # Incorrect dead air: aggregate
        total_unnecessary = sum(t.get("unnecessary_dead_air_ms", 0) for t in self._turns)
        summary["unnecessary_dead_air_ms"] = round(total_unnecessary)
        summary["pct_incorrect_dead_air"] = round(total_unnecessary / dead_air_ms * 100, 1) if dead_air_ms > 0 else 0

        # Call duration: first user_started to last bot_stopped
        first_user = next((t["user_started_at"] for t in self._turns if t.get("user_started_at")), None)
        last_bot = next((t["bot_stopped_at"] for t in reversed(self._turns) if t.get("bot_stopped_at")), None)
        if first_user and last_bot:
            summary["call_duration_s"] = round(last_bot - first_user, 2)
        else:
            summary["call_duration_s"] = None

        return summary

    def _flush(self):
        session = {
            "session_id": self._session_id,
            "mode": self._mode,
            "started_at": self._started_at,
            "ended_at": self._ended_at,
            "config": self._config,
            "turns": self._turns,
            "summary": self._build_summary(),
        }

        # Atomic write: temp file + rename
        fd, tmp_path = tempfile.mkstemp(
            dir=self._data_dir, suffix=".tmp", prefix=".session_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(session, f, indent=2)
            os.rename(tmp_path, self._output_path)
        except Exception:
            logger.exception("MetricsCollector: failed to write session JSON")
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

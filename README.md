# Gemini S2S vs Cascaded Pipeline

Two voice agents, same English use case, compared over real phone calls via Plivo.

- **Native S2S** (`agent_native.py`): one model end to end, **Gemini Live**. Audio in, audio out, no separate STT or TTS.
- **Cascaded** (`agent_cascaded.py`): three specialized models wired in sequence. Deepgram Nova-3 for speech to text, Google Gemini for the conversation, ElevenLabs for text to speech.

Live dashboard with call recordings, transcripts, and the full comparison: https://dashboard-s2s.vercel.app/

## How It Works

```
Native S2S
┌─────────┐     ┌─────────┐     ┌──────────────────┐
│  Phone  │────▶│  Plivo  │────▶│   Gemini Live    │
│  Call   │◀────│   WS    │◀────│  (audio in/out)  │
└─────────┘     └─────────┘     └──────────────────┘

Cascaded
┌─────────┐     ┌─────────┐     ┌──────────┐     ┌────────┐     ┌────────────┐
│  Phone  │────▶│  Plivo  │────▶│ Deepgram │────▶│ Gemini │────▶│ ElevenLabs │
│  Call   │◀────│   WS    │◀────│   STT    │     │  LLM   │     │    TTS     │
└─────────┘     └─────────┘     └──────────┘     └────────┘     └────────────┘
```

Both agents run on the same Pipecat framework over Plivo's bidirectional WebSocket transport. The only thing that changes between the two is what's inside the bot.

## Project Layout

```
agent_native.py       # Gemini Live (single-model S2S)
agent_cascaded.py     # Deepgram STT + Gemini LLM + ElevenLabs TTS
metrics_observer.py   # Per-turn pipeline waterfall capture
scripts/
  make_call.py        # Place an outbound test call via Plivo
dashboard/            # Static comparison dashboard (deployed to Vercel)
```

## Prerequisites

- Python 3.11+
- A [Plivo](https://www.plivo.com/) account with a phone number
- A Google Gemini API key (for both agents)
- A Deepgram API key (for the cascaded agent's STT)
- An ElevenLabs API key (for the cascaded agent's TTS)
- [ngrok](https://ngrok.com/) to expose the local server to Plivo

## Quick Start

1. **Clone and install**
   ```bash
   git clone https://github.com/renuyadav972/gemini-s2s.git
   cd gemini-s2s
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Fill in your Plivo, Google, Deepgram, and ElevenLabs credentials.

3. **Start ngrok**
   ```bash
   ngrok http 8000
   ```
   Put the ngrok HTTPS host (without `https://`) into `PUBLIC_DOMAIN` in `.env`.

4. **Run an agent**
   ```bash
   SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())") \
     python agent_native.py
   ```
   Or for the cascaded version:
   ```bash
   SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())") \
     python agent_cascaded.py
   ```

5. **Place a test call**
   ```bash
   python scripts/make_call.py --to +1XXXXXXXXXX
   ```

## What We Found

A few things stood out from running both side by side over real phone audio:

- **Language switching**: the native model handles multilingual code switching with no configuration. In a test call the caller mixed English, Spanish, and Hindi within the same sentence and Gemini Live matched the language of each question in its response. The cascaded version would need separate STT and TTS configured per language to do this.
- **Tone**: STT throws away how something was said and only passes text to the LLM. Gemini Live hears tone directly, which matters for agents that need to react to caller emotion.
- **Voice quality**: the native voice is generated as part of the reasoning, so pacing and emphasis match the intent. Most cascaded TTS sounds noticeably more bolted-on by comparison.
- **Trade off**: the cascaded pipeline lets you swap STT, LLM, or TTS independently. The native model is one black box. If something is off you wait on Google rather than swapping a component.

Full writeup with audio samples: https://dashboard-s2s.vercel.app/

## License

MIT

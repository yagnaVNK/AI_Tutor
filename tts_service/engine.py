"""TTS engine abstraction.

We support several engines (Edge-TTS, Qwen3-TTS, Vui) plus a tiny
sine-wave fallback for sanity checking. Edge-TTS is the default for
local dev because it's pure-Python, multilingual, free, and produces
natural-sounding audio over the network without any model download.

All engines stream raw 16-bit PCM mono frames at the configured sample
rate, which is what the voice gateway and browsers can play directly.
"""
from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import struct
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class BaseTTSEngine:
    sample_rate: int = 24000
    name: str = "base"

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        raise NotImplementedError


class SineWaveEngine(BaseTTSEngine):
    """CPU-only fallback that produces a quick 'beep' for the given text.

    This is intentionally tiny so the full system runs on dev machines
    without any GPU model download. It also makes the streaming
    contract obvious to anyone wiring up a new engine.
    """

    name = "sine"

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        seconds_per_word = 0.18
        word_count = max(1, len(text.split()))
        total_seconds = min(8.0, word_count * seconds_per_word)
        total_samples = int(total_seconds * self.sample_rate)
        frequency = 220.0 + (hash(text) % 220)
        chunk_samples = self.sample_rate // 20
        amplitude = 0.18

        for start in range(0, total_samples, chunk_samples):
            count = min(chunk_samples, total_samples - start)
            buf = bytearray()
            for i in range(count):
                t = (start + i) / self.sample_rate
                sample = amplitude * math.sin(2 * math.pi * frequency * t)
                buf += struct.pack("<h", int(sample * 32767))
            yield bytes(buf)


class Qwen3TTSEngine(BaseTTSEngine):
    """Placeholder for Qwen3-TTS integration.

    To enable, install the official package, load the model in __init__,
    and stream raw PCM frames from synthesize(). Until then we delegate
    to the SineWaveEngine so the rest of the system works.
    """

    name = "qwen3-tts"

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self._fallback = SineWaveEngine(sample_rate)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        async for chunk in self._fallback.synthesize(text):
            yield chunk


class VuiEngine(BaseTTSEngine):
    name = "vui"

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self._fallback = SineWaveEngine(sample_rate)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        async for chunk in self._fallback.synthesize(text):
            yield chunk


class EdgeTTSEngine(BaseTTSEngine):
    """Microsoft Edge online TTS via the `edge-tts` package.

    Streams MP3 chunks over WebSocket, which we decode in-process with
    PyAV (no external ffmpeg.exe needed) into 16-bit PCM mono at the
    configured sample rate. Multilingual out of the box — pick any voice
    from `edge-tts --list-voices`.
    """

    name = "edge-tts"

    DEFAULT_VOICE = "en-US-AriaNeural"

    def __init__(self, sample_rate: int = 24000, voice: str | None = None) -> None:
        self.sample_rate = sample_rate
        self.voice = voice or os.environ.get("TTS_VOICE") or self.DEFAULT_VOICE
        if self.voice == "default":
            self.voice = self.DEFAULT_VOICE

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        import edge_tts
        import av

        communicate = edge_tts.Communicate(text, self.voice)

        # Buffer MP3 chunks; PyAV needs a seekable container, so we wait
        # until the whole stream is collected and then decode + resample.
        # For typical sentence-sized inputs this introduces ~100-300ms of
        # extra latency per sentence, which is fine for a chat agent.
        mp3 = bytearray()
        async for event in communicate.stream():
            if event.get("type") == "audio":
                mp3.extend(event["data"])

        if not mp3:
            return

        loop = asyncio.get_running_loop()
        pcm_chunks = await loop.run_in_executor(
            None, self._mp3_to_pcm_chunks, bytes(mp3)
        )
        for chunk in pcm_chunks:
            yield chunk

    def _mp3_to_pcm_chunks(self, mp3_bytes: bytes) -> list[bytes]:
        import av

        container = av.open(io.BytesIO(mp3_bytes))
        try:
            stream = container.streams.audio[0]
        except IndexError:
            container.close()
            return []
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=self.sample_rate
        )
        chunks: list[bytes] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(bytes(resampled.to_ndarray().tobytes()))
        for resampled in resampler.resample(None):
            chunks.append(bytes(resampled.to_ndarray().tobytes()))
        container.close()
        return chunks


def load_engine() -> BaseTTSEngine:
    name = os.environ.get("TTS_ENGINE", "edge-tts").lower()
    sample_rate = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
    if name in ("edge", "edge-tts", "edge_tts"):
        return EdgeTTSEngine(sample_rate)
    if name == "qwen3-tts":
        return Qwen3TTSEngine(sample_rate)
    if name == "vui":
        return VuiEngine(sample_rate)
    return SineWaveEngine(sample_rate)

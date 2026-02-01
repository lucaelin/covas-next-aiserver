import time
from typing import AsyncGenerator
from cached_path import cached_path
import numpy as np
from kokoro_onnx import Kokoro
import samplerate

tts_model_names = [
    "hexgrad/Kokoro-82M",
]


def init_tts(asset: str = "hexgrad/Kokoro-82M"):
    if asset == "None":
        return

    model_path = cached_path(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx"
    ).as_posix()
    voices_path = cached_path(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    ).as_posix()

    kokoro = Kokoro(model_path, voices_path)

    # dry run
    print("Warming up Kokoro...")
    kokoro.create(
        "Hello World",
        voice="af_nova",
        speed=1.2,
        lang="en-us",
    )

    return (asset, kokoro)


async def tts(
    model: tuple[str, Kokoro], text: str, speed: float = 1.2, voice: str = "af"
) -> AsyncGenerator[tuple[np.ndarray, int], None]:
    start = time.time()

    if voice == "nova":
        voice = "af_nova"

    # Determine lang from voice
    lang = "en-us"
    if voice.startswith("af_") or voice.startswith("am_"):
        lang = "en-us"
    if voice.startswith("bf_") or voice.startswith("bm_"):
        lang = "en-gb"
    if voice.startswith("ef_") or voice.startswith("em_"):
        lang = "es"
    if voice.startswith("ff_") or voice.startswith("fm_"):
        lang = "fr-fr"
    if voice.startswith("hf_") or voice.startswith("hm_"):
        lang = "hi"
    if voice.startswith("if_") or voice.startswith("im_"):
        lang = "it"
    if voice.startswith("pf_") or voice.startswith("pm_"):
        lang = "pt-br"
    if voice.startswith("gf_") or voice.startswith("gm_"):
        lang = "de-de"

    stream = model[1].create_stream(
        text,
        voice=voice,
        speed=speed,
        lang=lang,
    )

    num_samples = 0
    async for samples, sample_rate in stream:
        samples = samplerate.resample(samples, 24000 / sample_rate, "sinc_best")
        sample_rate = 24000
        num_samples += len(samples)
        print(f"tts streaming: {len(samples)}")
        yield (samples, sample_rate)
    end = time.time()

    elapsed_seconds = end - start
    audio_duration = num_samples / 24000
    real_time_factor = elapsed_seconds / audio_duration

    print(f"TTS input: '{text}'")
    print(f"TTS Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"TTS Audio duration in seconds: {audio_duration:.3f}")
    print(
        f"TTS RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )


async def main():
    tts_model = init_tts("hexgrad/Kokoro-82M")
    text = "Hello, this is a test of the TTS system."
    async_generator = tts(tts_model, text, speed=1.0, voice="nova")
    async for audio, sample_rate in async_generator:
        print(f"Generated audio with {len(audio)} samples at {sample_rate} Hz")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

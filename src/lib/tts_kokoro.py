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
        # "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-quant.onnx"
        # "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-quant-convinteger.onnx"
    ).as_posix()
    voices_path = cached_path(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
    ).as_posix()

    # print voices
    with open(voices_path, "r") as f:
        import json

        voices = f.read()
        print(json.loads(voices).keys())

    kokoro = Kokoro(model_path, voices_path)

    # dry run
    print("Warming up Kokoro...")
    kokoro.create(
        "Hello World",
        voice="af",
        speed=1.2,
        lang="en-us",
    )

    return (asset, kokoro)


async def tts(
    model: tuple[str, Kokoro], text: str, speed: float = 1.2, voice: str = "af"
) -> AsyncGenerator[tuple[np.ndarray, int], None]:
    start = time.time()
    stream = model[1].create_stream(
        text,
        voice=voice,
        speed=speed,
        lang="en-us",
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

    print(f"The text is '{text}'")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")

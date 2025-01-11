import os
import platform
import sys
import time
from tracemalloc import start
from typing import Tuple
from cached_path import cached_path
import numpy as np
from kokoro_onnx import EspeakConfig, Kokoro
import samplerate

tts_model_names = [
    "hexgrad/Kokoro-82M",
]


def init_tts(asset: str = "hexgrad/Kokoro-82M"):
    if asset == "None":
        return

    model_path = cached_path(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
    ).as_posix()
    voices_path = cached_path(
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"
    ).as_posix()

    # print voices
    with open(voices_path, "r") as f:
        import json

        voices = f.read()
        print(json.loads(voices).keys())

    # find espeak lib and data path for pyinstaller in frozen mode
    if hasattr(sys, "frozen"):
        ext = (
            ".dll"
            if platform.system() == "Windows"
            else ".so" if platform.system() == "Linux" else ".dylib"
        )
        lib_name = (
            "espeak-ng" + ext
            if platform.system() == "Windows"
            else "libespeak-ng" + ext
        )
        espeak_lib_path = os.path.join(sys._MEIPASS, "espeakng_loader", lib_name)
        espeak_data_path = os.path.join(
            sys._MEIPASS, "espeakng_loader", "espeak-ng-data"
        )
        espeak_config = EspeakConfig(
            lib_path=espeak_lib_path, data_path=espeak_data_path
        )
        print("using bundled espeak-ng", espeak_lib_path, espeak_data_path)
        kokoro = Kokoro(model_path, voices_path, espeak_config)
    else:
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


def tts(model: Tuple[str, Kokoro], text: str, speed: float = 1.2, voice: str = "af"):
    start = time.time()
    samples, sample_rate = model[1].create(
        text,
        voice=voice,
        speed=speed,
        lang="en-us",
    )
    end = time.time()

    if len(samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        exit(1)

    elapsed_seconds = end - start
    audio_duration = len(samples) / sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    print(f"The text is '{text}'")
    print(f"Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"Audio duration in seconds: {audio_duration:.3f}")
    print(f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")

    # resample audio.samples to 24kHz to match openai
    samples = samplerate.resample(samples, 24000 / sample_rate, "sinc_best")
    sample_rate = 24000

    return (samples, sample_rate)

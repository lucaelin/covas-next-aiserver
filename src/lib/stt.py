import io
import time
from faster_whisper import WhisperModel
import samplerate
import soundfile as sf
import moonshine_onnx
import numpy as np
from functools import partial

stt_models_names = [
    "None",
    "distil-medium.en",
    "distil-small.en",
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "distil-large-v2",
    "distil-large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "moonshine/tiny",
    "moonshine/base",
]


def init_stt(model_name="distil-medium.en"):
    if model_name == "None":
        return None

    if model_name.startswith("moonshine/"):

        def transcribe(audio, language):
            text = moonshine_onnx.transcribe(audio, model=model_name)
            return text, {}

        return transcribe

    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe(audio, language):
        gen, info = model.transcribe(audio, language=language, beam_size=4)

        segments = []
        for segment in gen:
            segments.append(segment.text)

        return segments, info

    return transcribe


def stt(transcribe, wav: bytes, language="en-US"):
    # convert wav bytes to 16k S16_LE

    start = time.time()
    audio, rate = sf.read(io.BytesIO(wav))
    end = time.time()
    print("Read time:", end - start)
    start = time.time()
    audio = samplerate.resample(audio, 16000 / rate, "sinc_best")
    end = time.time()
    print("Resample time:", end - start)

    start = time.time()

    segments, info = transcribe(audio, language=language)

    end = time.time()
    print("Transcribe time:", end - start)

    return segments, info

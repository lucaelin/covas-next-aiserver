import io
import time
from numpy import ndarray
import samplerate
import soundfile as sf
import onnx_asr
from typing import Literal

stt_model_names = [
    "onnx_asr/nemo-parakeet-tdt-0.6b-v2",
]


def init_stt(model_name="onnx_asr/nemo-parakeet-tdt-0.6b-v2"):

    # strip onnx_asr/ from model_name if it exists
    if model_name.startswith("onnx_asr/"):
        model_name = model_name[len("onnx_asr/") :]

    # load model
    model = onnx_asr.load_model(model_name, quantization="int8")

    def transcribe(audio: ndarray, language: Literal["en", "de", "es", "fr"]):
        output = model.recognize(audio)
        return output

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

    # normalize audio
    audio = audio / max(abs(audio))

    # set dtype to float32
    audio = audio.astype("float32")

    start = time.time()
    output = transcribe(audio, language=language.split("-")[0])
    print(output)

    end = time.time()
    elapsed_seconds = end - start
    audio_duration = len(audio) / 16000
    real_time_factor = elapsed_seconds / audio_duration

    print(f"STT output: '{output}'")
    print(f"STT Audio duration in seconds: {audio_duration:.3f}")
    print(f"STT Elapsed seconds: {elapsed_seconds:.3f}")
    print(
        f"STT RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

    return [output], None

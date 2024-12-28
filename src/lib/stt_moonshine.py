import io
import time
import samplerate
import soundfile as sf
import moonshine_onnx

stt_model_names = [
    "moonshine/tiny",
    "moonshine/base",
]


def init_stt(model_name="moonshine/base"):

    def transcribe(audio, language):
        segments = moonshine_onnx.transcribe(audio, model=model_name)
        return [seg for seg in segments if seg], {}

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

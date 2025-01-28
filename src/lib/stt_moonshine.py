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
    elapsed_seconds = end - start
    audio_duration = len(audio) / 16000
    real_time_factor = elapsed_seconds / audio_duration

    print(f"STT output: '{''.join(segments)}'")
    print(f"STT Audio duration in seconds: {audio_duration:.3f}")
    print(f"STT Elapsed seconds: {elapsed_seconds:.3f}")
    print(
        f"STT RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

    return segments, info

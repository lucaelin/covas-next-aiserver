import io
import time
from cached_path import cached_path
from numpy import ndarray
import samplerate
import soundfile as sf
from typing import Literal
import sherpa_onnx

#from .stt_sherpa_canary import init, transcribe

stt_model_names = [
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2.tar.bz2",
    "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2",
    "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr.tar.bz2",
    "sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k-int8.tar.bz2",
    "sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2",
    #"sherpa-onnx-nemo-fast-conformer-transducer-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2",
]

def create_canary_recognizer(asset: str):
    path = cached_path(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + asset,
        extract_archive=True,
    )
    files = [file for file in path.glob("**/*")]

    encoder = [f for f in files if "encoder" in f.name][0]
    decoder = [f for f in files if "decoder" in f.name][0]
    tokens = [f for f in files if "tokens" in f.name][0]

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_canary(
        encoder=str(encoder),
        decoder=str(decoder),
        tokens=str(tokens),
        debug=True,
    )
    
    return recognizer

def create_transducer_recognizer(asset: str):
    path = cached_path(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + asset,
        extract_archive=True,
    )
    files = [file for file in path.glob("**/*")]

    encoder = [f for f in files if "encoder" in f.name][0]
    decoder = [f for f in files if "decoder" in f.name][0]
    joiner = [f for f in files if "joiner" in f.name][0]
    tokens = [f for f in files if "tokens" in f.name][0]

    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        tokens=str(tokens),
        model_type="nemo_transducer",
        debug=True,
    )

    return recognizer

def create_ctc_recognizer(asset: str):
    path = cached_path(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/" + asset,
        extract_archive=True,
    )
    files = [file for file in path.glob("**/*")]

    model = [f for f in files if "model" in f.name][0]
    tokens = [f for f in files if "tokens" in f.name][0]

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(model),
        tokens=str(tokens),
        debug=True,
    )

    return recognizer

def canary_decode(recognizer, samples, sample_rate, lang):
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)

    recognizer.recognizer.set_config(
        config=sherpa_onnx.OfflineRecognizerConfig(
            model_config=sherpa_onnx.OfflineModelConfig(
                canary=sherpa_onnx.OfflineCanaryModelConfig(
                    src_lang=lang,
                    tgt_lang=lang,
                )
            )
        )
    )

    recognizer.decode_stream(stream)
    return stream.result.text

def transducer_decode(recognizer, samples, sample_rate, lang):
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text

def ctc_decode(recognizer, samples, sample_rate, lang):
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text

def init_stt(model_name="sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2"):
    if model_name not in stt_model_names:
        raise ValueError(f"Model {model_name} not found in available models: {stt_model_names}")
    
    model = None
    if 'canary' in model_name:
        model = create_canary_recognizer(model_name)
    elif 'transducer' in model_name:
        model = create_transducer_recognizer(model_name)
    elif 'parakeet' in model_name:
        model = create_transducer_recognizer(model_name)
    elif '-ctc-' in model_name:
        model = create_ctc_recognizer(model_name)
    else:
        raise ValueError(f"Model type not recognized for {model_name}")
    
    def stt(audio: ndarray, language: Literal["en", "de", "es", "fr"]):
        if 'canary' in model_name:
            return canary_decode(model, audio, 16000, language)
        elif 'ctc' in model_name:
            return ctc_decode(model, audio, 16000, language)
        elif 'transducer' in model_name:
            return transducer_decode(model, audio, 16000, language)
        elif 'parakeet' in model_name:
            return transducer_decode(model, audio, 16000, language)
        else:
            raise ValueError(f"Model type not recognized for {model_name}")

    return model_name, stt


def stt(model, wav: bytes, language="en-US"):
    # convert wav bytes to 16k S16_LE
    start = time.time()
    audio, rate = sf.read(io.BytesIO(wav))
    end = time.time()
    print("Read time:", end - start)
    start = time.time()
    if audio.ndim > 1:
        audio = audio[:, 0]  # use only the first channel if stereo
    audio = samplerate.resample(audio, 16000 / rate, "sinc_best")
    end = time.time()
    print("Resample time:", end - start)

    # normalize audio
    audio = audio / max(abs(audio))

    start = time.time()
    output = model[1](audio, language=language.split("-")[0])

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
    
    # save audio to file for debugging
    #with open("debug_audio.wav", "wb") as f:
    #    sf.write(f, audio, 16000, format='WAV', subtype='PCM_16')

    return [output], None

if __name__ == "__main__":
    m = init_stt("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2",)
    with open("audio3.wav", "rb") as f:
        wav = f.read()
    text, _ = stt(m, wav, "en-US")
    print(text)
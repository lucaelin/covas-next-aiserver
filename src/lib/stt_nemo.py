import io
import time
from numpy import ndarray
import samplerate
import soundfile as sf
from nemo.collections.asr.models import EncDecMultiTaskModel
from typing import Literal

from transformers.models.opt.modeling_opt import OPTSdpaAttention

stt_model_names = [
    "nvidia/canary-180m-flash",
    "nvidia/canary-1b-flash",
]


def init_stt(model_name="nvidia/canary-180m-flash"):

    # load model
    nemo_model = EncDecMultiTaskModel.from_pretrained(model_name)
    if model_name in ["nvidia/canary-1b-flash", "nvidia/canary-180m-flash"]:
        # update decode params
        decode_cfg = nemo_model.cfg.decoding
        decode_cfg.beam.beam_size = 1
        nemo_model.change_decoding_strategy(decode_cfg)

    def transcribe(audio: ndarray, language: Literal["en", "de", "es", "fr"]):
        if model_name in ["nvidia/canary-1b-flash", "nvidia/canary-180m-flash"]:
            opts = {
                "audio": audio,
                "batch_size": 1,  # batch size to run the inference with
                "pnc": "yes",  # generate output with Punctuation and Capitalization
                "source_lang": language,
                "target_lang": language,
            }
        if model_name == "nvidia/parakeet-tdt-0.6b-v2":
            opts = {
                "audio": audio,
                "batch_size": 1,  # batch size to run the inference with
            }
        output = nemo_model.transcribe(**opts)
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

    start = time.time()
    output = transcribe(audio, language=language.split("-")[0])
    print(output[0])

    end = time.time()
    elapsed_seconds = end - start
    audio_duration = len(audio) / 16000
    real_time_factor = elapsed_seconds / audio_duration

    print(f"STT output: '{output[0].text}'")
    print(f"STT Audio duration in seconds: {audio_duration:.3f}")
    print(f"STT Elapsed seconds: {elapsed_seconds:.3f}")
    print(
        f"STT RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

    return [output[0].text], None

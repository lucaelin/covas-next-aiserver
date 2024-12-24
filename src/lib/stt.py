from faster_whisper import WhisperModel
from .stt_fasterwhisper import (
    init_stt as init_stt_whisper,
    stt as stt_whisper,
    stt_model_names as stt_model_names_whisper,
)
from .stt_moonshine import (
    init_stt as init_stt_moonshine,
    stt as stt_moonshine,
    stt_model_names as stt_model_names_moonshine,
)


stt_model_names = ["None"] + stt_model_names_whisper + stt_model_names_moonshine


def init_stt(model_name="None"):
    if model_name == "None":
        return None

    if model_name in stt_model_names_whisper:
        model = init_stt_whisper(model_name)
        return model

    if model_name in stt_model_names_moonshine:
        model = init_stt_moonshine(model_name)
        return model

    return None


def stt(model, wav: bytes, language="en-US"):

    if isinstance(model, WhisperModel):
        return stt_whisper(model, wav, language)
    else:
        return stt_moonshine(model, wav, language)

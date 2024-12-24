from faster_whisper import WhisperModel
from .stt_fasterwhisper import init_stt, stt, stt_model_names
from .stt_moonshine import (
    init_stt as init_moonshine,
    stt as stt_moonshine,
    stt_model_names as moonshine_model_names,
)


stt_model_names = ["None"] + stt_model_names + moonshine_model_names


def init_stt(model_name="None"):
    if model_name == "None":
        return None

    if model_name in stt_model_names:
        model = init_stt(model_name)
        return model

    if model_name in moonshine_model_names:
        model = init_moonshine(model_name)
        return model

    return None


def stt(model, wav: bytes, language="en-US"):

    if isinstance(model, WhisperModel):
        return stt(model, wav, language)
    else:
        return stt_moonshine(model, wav, language)

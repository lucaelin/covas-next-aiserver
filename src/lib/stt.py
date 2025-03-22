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
from .stt_nemo import (
    init_stt as init_stt_nemo,
    stt as stt_nemo,
    stt_model_names as stt_model_names_nemo,
)


stt_model_names = ["None"] + stt_model_names_nemo + stt_model_names_whisper + stt_model_names_moonshine


def init_stt(model_name="None"):
    if model_name == "None":
        return None
    
    if model_name in stt_model_names_nemo:
        model = init_stt_nemo(model_name)
        return ('nemo', model)

    if model_name in stt_model_names_whisper:
        model = init_stt_whisper(model_name)
        return ('whisper', model)

    if model_name in stt_model_names_moonshine:
        model = init_stt_moonshine(model_name)
        return ('moonshine', model)

    return None


def stt(model, wav: bytes, language="en-US"):

    if model[0] == 'whisper':
        return stt_whisper(model[1], wav, language)
    elif model[0] == 'moonshine':
        return stt_moonshine(model[1], wav, language)
    elif model[0] == 'nemo':
        return stt_nemo(model[1], wav, language)
    else:
        return None

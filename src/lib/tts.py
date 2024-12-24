from .tts_sherpa import (
    init_tts as init_tts_sherpa,
    tts as tts_sherpa,
    tts_model_names as tts_model_names_sherpa,
)

tts_model_names = ["None"] + tts_model_names_sherpa


def init_tts(model_name="None"):
    if model_name == "None":
        return None

    if model_name in tts_model_names:
        model = init_tts_sherpa(model_name)
        return model

    return None


def tts(model, input, speed, voice):
    return tts_sherpa(model, input, speed, voice)

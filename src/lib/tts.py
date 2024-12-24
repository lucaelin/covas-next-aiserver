from .tts_sherpa import init_tts, tts, tts_model_names

tts_model_names = ["None"] + tts_model_names


def init_tts(model_name="None"):
    if model_name == "None":
        return None

    if model_name in tts_model_names:
        model = init_tts(model_name)
        return model

    return None


def tts(model, prompt):
    return tts(model, prompt)

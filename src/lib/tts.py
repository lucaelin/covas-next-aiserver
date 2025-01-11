from typing import Any, Tuple
from .tts_sherpa import (
    init_tts as init_tts_sherpa,
    tts as tts_sherpa,
    tts_model_names as tts_model_names_sherpa,
)
from .tts_kokoro import (
    init_tts as init_tts_kokoro,
    tts as tts_kokoro,
    tts_model_names as tts_model_names_kokoro,
)

tts_model_names = ["None"] + tts_model_names_sherpa + tts_model_names_kokoro


def init_tts(model_name="None"):
    if model_name == "None":
        return (model_name, None)

    if model_name in tts_model_names_sherpa:
        model = init_tts_sherpa(model_name)
        return (model_name, model)

    if model_name in tts_model_names_kokoro:
        model = init_tts_kokoro(model_name)
        return (model_name, model)

    return None


def tts(model: Tuple[str, Any], input, speed, voice) -> Tuple[Any, int]:
    model_name = model[0]
    model = model[1]

    if model_name in tts_model_names_kokoro:
        return tts_kokoro(model, input, speed, voice)
    if model_name in tts_model_names_sherpa:
        return tts_sherpa(model, input, speed, voice)

    raise ValueError(f"Unknown TTS model: {model_name}")

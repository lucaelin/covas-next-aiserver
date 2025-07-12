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

# from .stt_nemo import (
#    init_stt as init_stt_nemo,
#    stt as stt_nemo,
#    stt_model_names as stt_model_names_nemo,
# )
from .stt_sherpa import (
    init_stt as init_stt_sherpa,
    stt as stt_sherpa,
    stt_model_names as stt_model_names_sherpa,
)
#from .stt_asr_onnx import (
#    init_stt as init_stt_asr_onnx,
#    stt as stt_asr_onnx,
#    stt_model_names as stt_model_names_asr_onnx,
#)


stt_model_names = (
    ["None"]
    #+ stt_model_names_asr_onnx
    + stt_model_names_sherpa
    #    + stt_model_names_nemo
    + stt_model_names_whisper
    + stt_model_names_moonshine
)


def init_stt(model_name="None"):
    if model_name == "None":
        return None
    
    if model_name in 'mix_multilingual':
        model_en = init_stt_sherpa('sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2')
        model_multi = init_stt_sherpa('sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k-int8.tar.bz2')
        return ("mix_multilingual", model_en, model_multi)

    # if model_name in stt_model_names_nemo:
    #    model = init_stt_nemo(model_name)
    #    return ("nemo", model)

    if model_name in stt_model_names_whisper:
        model = init_stt_whisper(model_name)
        return ("whisper", model)

    if model_name in stt_model_names_moonshine:
        model = init_stt_moonshine(model_name)
        return ("moonshine", model)

    #if model_name in stt_model_names_asr_onnx:
    #    model = init_stt_asr_onnx(model_name)
    #    return ("asr_onnx", model)
    
    if model_name in stt_model_names_sherpa:
        model = init_stt_sherpa(model_name)
        return ("sherpa", model)

    return None


def stt(model, wav: bytes, language="en-US"):
    if model[0] == "whisper":
        return stt_whisper(model[1], wav, language)
    elif model[0] == "moonshine":
        return stt_moonshine(model[1], wav, language)
    # elif model[0] == "nemo":
    #    return stt_nemo(model[1], wav, language)
    #elif model[0] == "asr_onnx":
    #    return stt_asr_onnx(model[1], wav, language)
    elif model[0] == "sherpa":
        return stt_sherpa(model[1], wav, language)
    elif model[0] == "mix_multilingual":
        if language in ["en", "en-US", "en-GB"]:
            return stt_sherpa(model[1], wav, language)
        else:
            return stt_sherpa(model[2], wav, language)
    else:
        return None

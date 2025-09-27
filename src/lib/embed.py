from llama_cpp import Llama
from .embed_ort import (
    init_embed as init_embed_ort,
    embed as embed_ort,
    embed_model_names as embed_model_names_ort,
)
from .embed_llamacpp import (
    init_embed as init_embed_llamacpp,
    embed as embed_llamacpp,
    embed_model_names as embed_model_names_llamacpp,
)


embed_model_names = ["None"] + embed_model_names_ort + embed_model_names_llamacpp


def init_embed(model_name="None"):
    if model_name == "None":
        return None

    if model_name in embed_model_names_ort:
        model = init_embed_ort(model_name)
        return model

    if model_name in embed_model_names_llamacpp:
        model = init_embed_llamacpp(model_name)
        return model

    return None


def embed(model, prompt):
    if isinstance(model, Llama):
        return embed_llamacpp(model, prompt)
    return embed_ort(model, prompt)

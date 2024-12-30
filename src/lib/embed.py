from llama_cpp import Llama
from .embed_jina import (
    init_embed as init_embed_jina,
    embed as embed_jina,
    embed_model_names as embed_model_names_jina,
)
from .embed_llamacpp import (
    init_embed as init_embed_llamacpp,
    embed as embed_llamacpp,
    embed_model_names as embed_model_names_llamacpp,
)


embed_model_names = ["None"] + embed_model_names_jina + embed_model_names_llamacpp


def init_embed(model_name="None"):
    if model_name == "None":
        return None

    if model_name in embed_model_names_jina:
        model = init_embed_jina(model_name)
        return model

    if model_name in embed_model_names_llamacpp:
        model = init_embed_llamacpp(model_name)
        return model

    return None


def embed(model, prompt):
    if isinstance(model, Llama):
        return embed_llamacpp(model, prompt)
    return embed_jina(model, prompt)

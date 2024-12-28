from .llm_llamacpp import (
    init_llm as init_llm_llama,
    llm as llm_llama,
    llm_model_names as llm_model_names_llama,
)

llm_model_names = ["None"] + llm_model_names_llama


def init_llm(model_name="None", use_disk_cache=False):
    if model_name == "None":
        return None

    if model_name in llm_model_names:
        model = init_llm_llama(model_name, use_disk_cache)
        return model

    return None


def llm(model, prompt):
    return llm_llama(model, prompt)

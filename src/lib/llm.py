from .llm_llamacpp import init_llm, llm, llm_model_names

llm_model_names = ["None"] + llm_model_names


def init_llm(model_name="None", use_disk_cache=False):
    if model_name == "None":
        return None

    if model_name in llm_model_names:
        model = init_llm(model_name, use_disk_cache)
        return model

    return None


def llm(model, prompt):
    return llm(model, prompt)

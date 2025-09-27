import json
import time
from typing import Any
from llama_cpp import Llama

model_presets: dict[str, dict[str, Any]] = {
    #"ggml-org/embeddinggemma-300M-GGUF": {
    #    "filename": "embeddinggemma-300M-Q8_0.gguf",
    #},
    "lmstudio-community/granite-embedding-107m-multilingual-GGUF": {
        "filename": "granite-embedding-107m-multilingual-Q8_0.gguf",
        "max_context": 512 * 16,
        "rope_scaling_type": 1,
        "rope_freq_scale": 16,
    },
    "lmstudio-community/granite-embedding-278m-multilingual-GGUF": {
        "filename": "granite-embedding-278m-multilingual-Q8_0.gguf",
        "max_context": 512 * 16,
        "rope_scaling_type": 1,
        "rope_freq_scale": 16,
    },
}

embed_model_names = list(model_presets.keys())


def init_embed(model_path: str) -> Llama:
    model_preset = model_presets.get(model_path)
    if not model_preset:
        raise ValueError(f"Unknown model: {model_path}")
    llm = Llama.from_pretrained(
        repo_id=model_path,
        filename=model_preset.get("filename"),
        n_ctx=model_preset.get("max_context", 8 * 1024),
        n_gpu_layers=1000,
        embedding=True,
    )

    return llm


def embed(model: Llama, prompt):
    response = model.create_embedding(prompt.get("input"))

    return response


if __name__ == "__main__":
    model = init_embed("lmstudio-community/granite-embedding-278m-multilingual-GGUF")
    prompt = {"input": "Hello, world!"}
    start = time.time()
    embd = embed(model, prompt)
    end = time.time()
    print(json.dumps(embd))
    print(f"Time taken: {end - start} seconds")

    prompt = {"input": "This statement is completely nonsensical."}
    start = time.time()
    embd = embed(model, prompt)
    end = time.time()
    print(json.dumps(embd))
    print(f"Time taken: {end - start} seconds")

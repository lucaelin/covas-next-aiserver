import time
from tracemalloc import start
from typing import TypedDict
from cached_path import cached_path
import onnxruntime
import numpy as np
from transformers import AutoTokenizer, PretrainedConfig

embed_model_names = [
    "onnx-community/embeddinggemma-300m-ONNX",
    "jina-embeddings-v3",
]

class EmbeddingResultEmbedding(TypedDict):
    embedding: list[float]

class EmbeddingResult(TypedDict):
    model: str
    data: list[EmbeddingResultEmbedding]

# Mean pool function
def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray):
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask


def init_embed(model_name="onnx-community/embeddinggemma-300m-ONNX"):
    # Load model
    if model_name == "jina-embeddings-v3":
        model_path = cached_path(
            "https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model_fp16.onnx?download=true",
            extract_archive=False,
        )

        # Load tokenizer and model config
        tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
        config = PretrainedConfig.from_pretrained("jinaai/jina-embeddings-v3")

        # ONNX session
        session = onnxruntime.InferenceSession(model_path)
        
        def get_embedding(text: str) -> EmbeddingResult:
            # Tokenize input
            input_text = tokenizer(text, return_tensors="np")

            # Task type
            # retrieval.query: Used for query embeddings in asymmetric retrieval tasks
            # retrieval.passage: Used for passage embeddings in asymmetric retrieval tasks
            # separation: Used for embeddings in clustering and re-ranking applications
            # classification: Used for embeddings in classification tasks
            # text-matching: Used for embeddings in tasks that quantify similarity between two texts, such as STS or symmetric retrieval tasks
            task_type = "text-matching"
            task_id = np.array(config.lora_adaptations.index(task_type), dtype=np.int64)

            # Prepare inputs for ONNX model
            inputs = {
                "input_ids": input_text["input_ids"],
                "attention_mask": input_text["attention_mask"],
                "task_id": task_id,
            }

            # Run model
            outputs = session.run(None, inputs)[0]

            # Apply mean pooling and normalization to the model outputs
            embeddings = mean_pooling(outputs, input_text["attention_mask"])
            embeddings = embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

            return {
                "model": model_name,
                "data": [
                    {
                        "embedding": embeddings.tolist(),
                    }
                ]
            }

        return get_embedding
    if model_name == "onnx-community/embeddinggemma-300m-ONNX":
        model_path = cached_path(
            f"hf://onnx-community/embeddinggemma-300m-ONNX",
        )

        # Load tokenizer and model config
        tokenizer = AutoTokenizer.from_pretrained("onnx-community/embeddinggemma-300m-ONNX")
        config = PretrainedConfig.from_pretrained("onnx-community/embeddinggemma-300m-ONNX")

        # ONNX session
        session = onnxruntime.InferenceSession(model_path.as_posix()+"/onnx/model_fp16.onnx")
        
        def get_embedding(text: str) -> EmbeddingResult:
            prefixes = {
                "query": "task: search result | query: ",
                "document": "title: none | text: ",
            }

            inputs = tokenizer([prefixes["document"] + text], padding=True, return_tensors="np")

            _, sentence_embedding = session.run(None, inputs.data)
            
            return {
                "model": model_name,
                "data": [
                    {
                        "embedding": sentence_embedding[0].tolist()
                    }
                ]
            }
        return get_embedding

    return None


def embed(model, prompt):
    return model(prompt.get("input"))

if __name__ == "__main__":
    model = init_embed('onnx-community/embeddinggemma-300m-ONNX')

    prompt = {"input": "Hello, world!"}
    start = time.time()
    embd = embed(model, prompt)
    end = time.time()
    print(embd)
    print(f"Time taken: {end - start} seconds")

    prompt = {"input": "This statement is completely nonsensical."}
    start = time.time()
    embd = embed(model, prompt)
    end = time.time()
    print(embd)
    print(f"Time taken: {end - start} seconds")

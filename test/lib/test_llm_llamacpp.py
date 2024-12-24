from llama_cpp import Llama
from src.lib.llm_llamacpp import init_llm, llm, llm_model_names


def test_model_list():
    """Test the model list is a list of strings"""
    assert isinstance(llm_model_names, list)
    assert all(isinstance(model, str) for model in llm_model_names)


def test_llm():
    """Test that the LLM model can be initialized using lmstudio-community/Llama-3.2-1B-Instruct-GGUF"""
    model = init_llm("lmstudio-community/Llama-3.2-1B-Instruct-GGUF")
    assert model is not None
    assert isinstance(model, Llama)

    """ test that the model can generate text """
    prompt = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 1,
    }
    result = llm(model, prompt)
    assert result is not None
    assert isinstance(result["choices"][0]["message"]["content"], str)
    assert len(result["choices"][0]["message"]["content"]) > 0
    assert result["usage"]["prompt_tokens"] == 16
    assert result["usage"]["completion_tokens"] == 1

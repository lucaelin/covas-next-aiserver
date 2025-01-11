from random import sample
from kokoro_onnx import Kokoro
from src.lib.tts_kokoro import init_tts, tts, tts_model_names
import pytest


def test_model_list():
    """Test the model list is a list of strings"""
    assert isinstance(tts_model_names, list)
    assert all(isinstance(model, str) for model in tts_model_names)


@pytest.mark.asyncio
async def test_tts():
    """Test that the TTS model can be initialized using hexgrad/Kokoro-82M"""
    model = init_tts("hexgrad/Kokoro-82M")
    assert model[1] is not None
    assert isinstance(model[1], Kokoro)

    """ test that the model can generate audio """
    prompt = "hello there"
    stream = tts(model, prompt)
    audio = []
    samplerate = 0
    async for samples, rate in stream:
        audio.extend(samples)
        samplerate = rate

    assert samplerate == 24000
    assert len(audio) > 12000  # at least 0.5 seconds
    # not all zeros
    assert any([x != 0 for x in audio])

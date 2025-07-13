from sherpa_onnx import OfflineTts
from src.lib.tts_sherpa_vits import init_tts, tts, tts_model_names
import pytest


def test_model_list():
    """Test the model list is a list of strings"""
    assert isinstance(tts_model_names, list)
    assert all(isinstance(model, str) for model in tts_model_names)


@pytest.mark.asyncio
async def test_tts():
    """Test that the TTS model can be initialized using vits-piper-en_US-ljspeech-high.tar.bz2"""
    model = init_tts("vits-piper-en_US-ljspeech-high-fp16.tar.bz2")
    assert model is not None
    assert isinstance(model, OfflineTts)

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

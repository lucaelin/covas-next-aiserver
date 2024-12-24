from faster_whisper import WhisperModel
from src.lib.stt_fasterwhisper import init_stt, stt, stt_model_names


def test_model_list():
    """Test the model list is a list of strings"""
    assert isinstance(stt_model_names, list)
    assert all(isinstance(model, str) for model in stt_model_names)


def test_tts():
    """Test that the TTS model can be initialized using tiny.en"""
    model = init_stt("tiny.en")
    assert model is not None
    assert isinstance(model, WhisperModel)

    """ test that the model can transcribe audio """
    wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    segments, info = stt(model, wav)
    assert isinstance(segments, list)
    assert len(segments) == 0

import moonshine_onnx
from src.lib.stt_moonshine import init_stt, stt, stt_model_names


def test_model_list():
    """Test the model list is a list of strings"""
    assert isinstance(stt_model_names, list)
    assert all(isinstance(model, str) for model in stt_model_names)


def test_tts():
    """Test that the TTS model can be initialized using moonshine/tiny"""
    model = init_stt("moonshine/tiny")
    assert model is not None
    # assert that the model is a function
    assert callable(model)

    """ test that the model can transcribe audio """

    # simulate a wav file with 1 second of audio and a correct header for 16kHz mono PCM 1s
    wav = (
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data"
        + b"\x01" * 16000
    )
    segments, info = stt(model, wav)
    assert isinstance(segments, list)
    assert len(segments) == 0

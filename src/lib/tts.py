from typing import Any, AsyncGenerator, Tuple
import io
import soundfile as sf
from .tts_sherpa import (
    init_tts as init_tts_sherpa,
    tts as tts_sherpa,
    tts_model_names as tts_model_names_sherpa,
)
from .tts_kokoro import (
    init_tts as init_tts_kokoro,
    tts as tts_kokoro,
    tts_model_names as tts_model_names_kokoro,
)

tts_model_names = ["None"] + tts_model_names_sherpa + tts_model_names_kokoro


def init_tts(model_name="None"):
    if model_name == "None":
        return (model_name, None)

    if model_name in tts_model_names_sherpa:
        model = init_tts_sherpa(model_name)
        return (model_name, model)

    if model_name in tts_model_names_kokoro:
        model = init_tts_kokoro(model_name)
        return (model_name, model)

    return None


async def audio_stream_generator(stream, response_format):
    if response_format not in ["wav", "raw", "pcm"]:
        raise ValueError("Invalid response_format")

    buffer = io.BytesIO()
    buffer.name = "audio." + response_format
    with sf.SoundFile(
        buffer,
        mode="w",
        channels=1,
        samplerate=24000,
        subtype="PCM_16",
        format="WAV" if response_format == "wav" else "RAW",
    ) as wfile:
        buffer.seek(0)
        data = buffer.read()
        yield data
        async for chunk, samplerate in stream:
            # print("chunk:", len(chunk))
            current_pos = buffer.tell()
            # print("current_pos:", current_pos)
            wfile.write(chunk)
            buffer.seek(current_pos)
            data = buffer.read()
            # print(f"audio_stream_generator: {len(data)}")
            yield data


async def tts(
    model: Tuple[str, Any], input, speed, voice, response_format="wav"
) -> AsyncGenerator[bytes, Any]:
    model_name = model[0]
    model = model[1]

    if model_name in tts_model_names_kokoro:
        stream = tts_kokoro(model, input, speed, voice)
        return audio_stream_generator(stream, response_format)
    if model_name in tts_model_names_sherpa:
        stream = tts_sherpa(model, input, speed, voice)
        return audio_stream_generator(stream, response_format)

    raise ValueError(f"Unknown TTS model: {model_name}")

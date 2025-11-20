from typing import Any, AsyncGenerator, Tuple
import io
import soundfile as sf
import numpy as np
from .tts_sherpa_vits import (
    init_tts as init_tts_sherpa,
    tts as tts_sherpa,
    tts_model_names as tts_model_names_sherpa,
)
from .tts_kokoro import (
    init_tts as init_tts_kokoro,
    tts as tts_kokoro,
    tts_model_names as tts_model_names_kokoro,
)
from .tts_kitten import (
    init_tts as init_tts_kitten,
    tts as tts_kitten,
    tts_model_names as tts_model_names_kitten,
)
from .tts_supertonic import (
    init_tts as init_tts_supertonic,
    tts as tts_supertonic,
    tts_model_names as tts_model_names_supertonic,
)

tts_model_names = ["None"] + tts_model_names_kokoro + tts_model_names_supertonic + tts_model_names_kitten + tts_model_names_sherpa


def init_tts(model_name="None"):
    if model_name == "None":
        return (model_name, None)

    if model_name in tts_model_names_sherpa:
        model = init_tts_sherpa(model_name)
        return (model_name, model)

    if model_name in tts_model_names_kokoro:
        model = init_tts_kokoro(model_name)
        return (model_name, model)

    if model_name in tts_model_names_kitten:
        model = init_tts_kitten(model_name)
        return (model_name, model)
    
    if model_name in tts_model_names_supertonic:
        model = init_tts_supertonic(model_name)
        return (model_name, model)

    return (model_name, None)


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
            if len(chunk):
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

    if input.strip() == "":
        print("Empty input, returning empty stream")

        async def _empty_input_async_generator():
            # This async generator yields the single item expected by audio_stream_generator
            yield (b"", 24000)

        return audio_stream_generator(_empty_input_async_generator(), response_format)

    if model_name in tts_model_names_kokoro:
        stream = tts_kokoro(model, input, speed, voice)
        return audio_stream_generator(stream, response_format)
    if model_name in tts_model_names_sherpa:
        stream = tts_sherpa(model, input, speed, voice)
        return audio_stream_generator(stream, response_format)
    if model_name in tts_model_names_kitten:
        stream = tts_kitten(model, input, speed, voice)
        return audio_stream_generator(stream, response_format)
    if model_name in tts_model_names_supertonic:
        stream = tts_supertonic(model, input, speed, voice)
        return audio_stream_generator(stream, response_format)

    raise ValueError(f"Unknown TTS model: {model_name}")

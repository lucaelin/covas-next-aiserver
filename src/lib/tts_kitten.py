from kittentts import KittenTTS

from typing import AsyncGenerator

import time
import numpy as np
import samplerate

tts_model_names = [
    "KittenML/kitten-tts-nano-0.1",
]


def init_tts(asset: str = "KittenML/kitten-tts-nano-0.1"):
    m = KittenTTS(asset)
    return (asset, m)



async def tts(
    model: tuple[str, KittenTTS], text: str, speed: float = 1.2, voice: str = "expr-voice-2-f"
) -> AsyncGenerator[tuple[np.ndarray, int], None]:
    m = model[1]
    start = time.time()
    
    if voice == "nova":
        voice = "expr-voice-2-f"
    audio = m.generate(text, voice=voice, speed=speed)
    # available_voices : [  'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',  'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f' ]

    # sample_rate = 24000 # already set in the model
    # audio = samplerate.resample(audio, 24000 / sample_rate, "sinc_best")
    
    end = time.time()

    num_samples = audio.shape[0]
    elapsed_seconds = end - start
    audio_duration = num_samples / 24000
    real_time_factor = elapsed_seconds / audio_duration

    print(f"TTS input: '{text}'")
    print(f"TTS Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"TTS Audio duration in seconds: {audio_duration:.3f}")
    print(
        f"TTS RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )
    yield (audio, 24000)  # Assuming the model outputs audio at 24000 Hz
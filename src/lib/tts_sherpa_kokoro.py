import time
from cached_path import cached_path
import numpy as np
import sherpa_onnx
import samplerate

tts_model_names = [
    "kokoro-int8-multi-lang-v1_0.tar.bz2",
    "kokoro-multi-lang-v1_0.tar.bz2",
    "kokoro-int8-multi-lang-v1_1.tar.bz2",
    "kokoro-multi-lang-v1_1.tar.bz2",
    "kokoro-int8-en-v0_19.tar.bz2",
    "kokoro-en-v0_19.tar.bz2",
]


def init_tts(asset: str = "kokoro-int8-multi-lang-v1_0.tar.bz2"):
    if asset == "None":
        return

    path = cached_path(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/" + asset,
        extract_archive=True,
    )
    files = [file for file in path.glob("*/*")]
    #print('model downloaded files: ', files)

    kokoro_model = [f for f in files if "model" in f.name][0]
    kokoro_voices = [f for f in files if "voices" in f.name][0]
    kokoro_tokens = [f for f in files if "tokens" in f.name][0]
    kokoro_data_dir = [f for f in files if "espeak-ng-data" in f.name][0]
    kokoro_dict_dir = [f for f in files if "dict" in f.name][0]
    kokoro_lexicon = [f for f in files if "lexicon-us-en" in f.name][0]
    
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(kokoro_model),
                voices=str(kokoro_voices),
                tokens=str(kokoro_tokens),
                data_dir=str(kokoro_data_dir),
                dict_dir=str(kokoro_dict_dir),
                lexicon=str(kokoro_lexicon),
            ),
            provider="cpu",
            debug=False,
            num_threads=8,
        ),
        rule_fsts="",
        max_num_sentences=1,
    )
    print(tts_config)
    if not tts_config.validate():
        raise ValueError("Please check your config")

    tts = sherpa_onnx.OfflineTts(tts_config)
    return tts


async def tts(
    model: sherpa_onnx.OfflineTts, text: str, speed: float = 1.0, voice: str = "nova"
):
    def generated_audio_callback(samples: np.ndarray, progress: float):
        print(f"Generated audio with {len(samples)} samples, progress: {progress:.2f}")
        # 1 means to keep generating
        # 0 means to stop generating
        return 1

    start = time.time()
    sid = {"": 0, "nova": 0}[voice]
    if sid is None:
        raise ValueError(f"Unknown voice: {voice}")
    audio = model.generate(text, sid=0, speed=speed, callback=generated_audio_callback)
    end = time.time()

    if len(audio.samples) == 0:
        print("Error in generating audios. Please read previous error messages.")
        exit(1)

    num_samples = 0

    samples = samplerate.resample(audio.samples, 24000 / audio.sample_rate, "sinc_best")
    sample_rate = 24000
    num_samples += len(samples)
    yield (samples, sample_rate)

    end = time.time()

    elapsed_seconds = end - start
    audio_duration = num_samples / 24000
    real_time_factor = elapsed_seconds / audio_duration

    print(f"TTS input: '{text}'")
    print(f"TTS Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"TTS Audio duration in seconds: {audio_duration:.3f}")
    print(
        f"TTS RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

async def main():
    tts_model = init_tts("kokoro-multi-lang-v1_0.tar.bz2")
    text = "Hi"
    async_generator = tts(tts_model, text, speed=1.0, voice="nova")
    async for audio, sample_rate in async_generator:
        print(f"Generated audio with {len(audio)} samples at {sample_rate} Hz")
        
    text = "Hello, this is a test of the TTS system."
    async_generator = tts(tts_model, text, speed=1.0, voice="nova")
    async for audio, sample_rate in async_generator:
        print(f"Generated audio with {len(audio)} samples at {sample_rate} Hz")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
import json
import os
import time
import re
from contextlib import contextmanager
from typing import Optional, AsyncGenerator
from cached_path import cached_path
from unicodedata import normalize
import numpy as np
import onnxruntime as ort
import samplerate

# --- helper.py content start ---

class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path, "r") as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str) -> str:
        # TODO: add more preprocessing
        text = normalize("NFKD", text)
        return text

    def _get_text_mask(self, text_ids_lengths: np.ndarray) -> np.ndarray:
        text_mask = length_to_mask(text_ids_lengths)
        return text_mask

    def _text_to_unicode_values(self, text: str) -> np.ndarray:
        unicode_values = np.array(
            [ord(char) for char in text], dtype=np.uint16
        )  # 2 bytes
        return unicode_values

    def __call__(self, text_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        text_list = [self._preprocess_text(t) for t in text_list]
        text_ids_lengths = np.array([len(text) for text in text_list], dtype=np.int64)
        text_ids = np.zeros((len(text_list), text_ids_lengths.max()), dtype=np.int64)
        for i, text in enumerate(text_list):
            unicode_vals = self._text_to_unicode_values(text)
            text_ids[i, : len(unicode_vals)] = np.array(
                [self.indexer[val] for val in unicode_vals], dtype=np.int64
            )
        text_mask = self._get_text_mask(text_ids_lengths)
        return text_ids, text_mask


class Style:
    def __init__(self, style_ttl_onnx: np.ndarray, style_dp_onnx: np.ndarray):
        self.ttl = style_ttl_onnx
        self.dp = style_dp_onnx


class TextToSpeech:
    def __init__(
        self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        dp_ort: ort.InferenceSession,
        text_enc_ort: ort.InferenceSession,
        vector_est_ort: ort.InferenceSession,
        vocoder_ort: ort.InferenceSession,
    ):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def sample_noisy_latent(
        self, duration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = ((wav_len_max + chunk_size - 1) / chunk_size).astype(np.int32)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy_latent = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        latent_mask = get_latent_mask(
            wav_lengths, self.base_chunk_size, self.chunk_compress_factor
        )
        noisy_latent = noisy_latent * latent_mask
        return noisy_latent, latent_mask

    def _infer(
        self, text_list: list[str], style: Style, total_step: int, speed: float = 1.05
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            len(text_list) == style.ttl.shape[0]
        ), "Number of texts must match number of style vectors"
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list)
        dur_onnx, *_ = self.dp_ort.run(
            None, {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask}
        )
        dur_onnx = dur_onnx / speed
        text_emb_onnx, *_ = self.text_enc_ort.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        )  # dur_onnx: [bsz]
        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(
                None,
                {
                    "noisy_latent": xt,
                    "text_emb": text_emb_onnx,
                    "style_ttl": style.ttl,
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": current_step,
                    "total_step": total_step_np,
                },
            )
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        return wav, dur_onnx

    def __call__(
        self,
        text: str,
        style: Style,
        total_step: int,
        speed: float = 1.05,
        silence_duration: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            style.ttl.shape[0] == 1
        ), "Single speaker text to speech only supports single style"
        text_list = chunk_text(text)
        wav_cat = None
        dur_cat = None
        for text in text_list:
            wav, dur_onnx = self._infer([text], style, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur_onnx
            else:
                silence = np.zeros(
                    (1, int(silence_duration * self.sample_rate)), dtype=np.float32
                )
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat += dur_onnx + silence_duration
        return wav_cat, dur_cat

    def batch(
        self, text_list: list[str], style: Style, total_step: int, speed: float = 1.05
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._infer(text_list, style, total_step, speed)


def length_to_mask(lengths: np.ndarray, max_len: Optional[int] = None) -> np.ndarray:
    """
    Convert lengths to binary mask.

    Args:
        lengths: (B,)
        max_len: int

    Returns:
        mask: (B, 1, max_len)
    """
    max_len = max_len or lengths.max()
    ids = np.arange(0, max_len)
    mask = (ids < np.expand_dims(lengths, axis=1)).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def get_latent_mask(
    wav_lengths: np.ndarray, base_chunk_size: int, chunk_compress_factor: int
) -> np.ndarray:
    latent_size = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + latent_size - 1) // latent_size
    latent_mask = length_to_mask(latent_lengths)
    return latent_mask


def load_onnx(
    onnx_path: str, opts: ort.SessionOptions, providers: list[str]
) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)


def load_onnx_all(
    paths: dict[str, str], opts: ort.SessionOptions, providers: list[str]
) -> tuple[
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
    ort.InferenceSession,
]:
    dp_ort = load_onnx(paths["duration_predictor"], opts, providers)
    text_enc_ort = load_onnx(paths["text_encoder"], opts, providers)
    vector_est_ort = load_onnx(paths["vector_estimator"], opts, providers)
    vocoder_ort = load_onnx(paths["vocoder"], opts, providers)
    return dp_ort, text_enc_ort, vector_est_ort, vocoder_ort


def load_cfgs(cfg_path: str) -> dict:
    with open(cfg_path, "r") as f:
        cfgs = json.load(f)
    return cfgs


def load_text_processor(indexer_path: str) -> UnicodeProcessor:
    text_processor = UnicodeProcessor(indexer_path)
    return text_processor


def load_text_to_speech(paths: dict[str, str], use_gpu: bool = False) -> TextToSpeech:
    opts = ort.SessionOptions()
    if use_gpu:
        # raise NotImplementedError("GPU mode is not fully tested")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Using GPU for inference")
    else:
        providers = ["CPUExecutionProvider"]
        print("Using CPU for inference")
    cfgs = load_cfgs(paths["config"])
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = load_onnx_all(
        paths, opts, providers
    )
    text_processor = load_text_processor(paths["indexer"])
    return TextToSpeech(
        cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort
    )


def load_voice_style(voice_style_paths: list[str], verbose: bool = False) -> Style:
    bsz = len(voice_style_paths)

    # Read first file to get dimensions
    with open(voice_style_paths[0], "r") as f:
        first_style = json.load(f)
    ttl_dims = first_style["style_ttl"]["dims"]
    dp_dims = first_style["style_dp"]["dims"]

    # Pre-allocate arrays with full batch size
    ttl_style = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp_style = np.zeros([bsz, dp_dims[1], dp_dims[2]], dtype=np.float32)

    # Fill in the data
    for i, voice_style_path in enumerate(voice_style_paths):
        with open(voice_style_path, "r") as f:
            voice_style = json.load(f)

        ttl_data = np.array(
            voice_style["style_ttl"]["data"], dtype=np.float32
        ).flatten()
        ttl_style[i] = ttl_data.reshape(ttl_dims[1], ttl_dims[2])

        dp_data = np.array(voice_style["style_dp"]["data"], dtype=np.float32).flatten()
        dp_style[i] = dp_data.reshape(dp_dims[1], dp_dims[2])

    if verbose:
        print(f"Loaded {bsz} voice styles")
    return Style(ttl_style, dp_style)


@contextmanager
def timer(name: str):
    start = time.time()
    print(f"{name}...")
    yield
    print(f"  -> {name} completed in {time.time() - start:.2f} sec")


def sanitize_filename(text: str, max_len: int) -> str:
    """Sanitize filename by replacing non-alphanumeric characters with underscores"""
    prefix = text[:max_len]
    return re.sub(r"[^a-zA-Z0-9]", "_", prefix)


def chunk_text(text: str, max_len: int = 300) -> list[str]:
    """
    Split text into chunks by paragraphs and sentences.

    Args:
        text: Input text to chunk
        max_len: Maximum length of each chunk (default: 300)

    Returns:
        List of text chunks
    """
    # Split by paragraph (two or more newlines)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]

    chunks = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Split by sentence boundaries (period, question mark, exclamation mark followed by space)
        # But exclude common abbreviations like Mr., Mrs., Dr., etc. and single capital letters like F.
        pattern = r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
        sentences = re.split(pattern, paragraph)

        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks

# --- helper.py content end ---

# --- Module API ---

tts_model_names = [
    "supertonic-v1",
]

def init_tts(asset: str = "supertonic-v1"):
    if asset == "None":
        return None
    
    base_url = "https://huggingface.co/Supertone/supertonic/resolve/main/onnx"
    files = {
        "duration_predictor": "duration_predictor.onnx",
        "text_encoder": "text_encoder.onnx",
        "vector_estimator": "vector_estimator.onnx",
        "vocoder": "vocoder.onnx",
        "config": "tts.json",
        "indexer": "unicode_indexer.json"
    }
    
    paths = {}
    print("Downloading/Loading Supertonic models...")
    try:
        for key, filename in files.items():
            url = f"{base_url}/{filename}"
            paths[key] = cached_path(url).as_posix()
    except Exception as e:
        print(f"Failed to download models: {e}")
        return None

    voice_base_url = "https://huggingface.co/Supertone/supertonic/resolve/main/voice_styles"
    voice_files = ["F1.json", "F2.json", "M1.json", "M2.json"]
    voices = {}
    print("Downloading/Loading Supertonic voices...")
    try:
        for filename in voice_files:
            url = f"{voice_base_url}/{filename}"
            voice_name = filename.replace(".json", "")
            voices[voice_name] = cached_path(url).as_posix()
    except Exception as e:
        print(f"Failed to download voices: {e}")
        return None

    # Check for GPU availability
    use_gpu = False
    if "CUDAExecutionProvider" in ort.get_available_providers():
        use_gpu = True

    try:
        model = load_text_to_speech(paths, use_gpu=use_gpu)
        return (asset, model, voices)
    except Exception as e:
        print(f"Failed to load Supertonic model: {e}")
        return None


async def tts(
    model: tuple[str, TextToSpeech, dict[str, str]], text: str, speed: float = 1.0, voice: str = "M1"
) -> AsyncGenerator[tuple[np.ndarray, int], None]:
    start = time.time()
    
    if model is None:
        return

    asset_path, tts_engine, voices = model
    
    # voice argument is treated as voice name or path to style json
    voice_path = None
    
    # Check if voice is in pre-downloaded voices
    if voice in voices:
        voice_path = voices[voice]
    else:
        raise ValueError(f"Voice {voice} not found in available voices: {list(voices.keys())}")

    try:
        style = load_voice_style([voice_path])
    except Exception as e:
        print(f"Failed to load voice style {voice_path}: {e}")
        return

    text_list = chunk_text(text)
    
    total_step = 5 # Default from example
    silence_duration = 0.3
    
    num_samples = 0
    for i, chunk in enumerate(text_list):
        # tts_engine._infer returns (wav, dur_onnx)
        # wav shape is (1, samples)
        wav, dur_onnx = tts_engine._infer([chunk], style, total_step, speed)
        
        samples = wav.flatten()
        sample_rate = tts_engine.sample_rate

        if sample_rate != 24000:
            samples = samplerate.resample(samples, 24000 / sample_rate, "sinc_best")
            sample_rate = 24000
        
        num_samples += len(samples)
        print(f"tts streaming: {len(samples)}")
        yield (samples, sample_rate)
        
        # Add silence between chunks if not the last one
        if i < len(text_list) - 1:
             silence_samples = int(silence_duration * sample_rate)
             silence = np.zeros(silence_samples, dtype=np.float32)
             num_samples += len(silence)
             yield (silence, sample_rate)

    end = time.time()

    elapsed_seconds = end - start
    audio_duration = num_samples / 24000
    real_time_factor = elapsed_seconds / audio_duration if audio_duration > 0 else 0

    print(f"TTS input: '{text}'")
    print(f"TTS Elapsed seconds: {elapsed_seconds:.3f}")
    print(f"TTS Audio duration in seconds: {audio_duration:.3f}")
    print(
        f"TTS RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}"
    )

async def main():
    # Example usage
    tts_model = init_tts()
    if tts_model is None:
        return

    text = "Hello, this is a test of the Supertonic TTS system."
    async_generator = tts(tts_model, text, speed=1.0, voice="M1")
    
    async for audio, sample_rate in async_generator:
        print(f"Generated audio with {len(audio)} samples at {sample_rate} Hz")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

import json
from typing import TypedDict
import os
import sys

from pick import pick
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from lib.embed import init_embed, embed, embed_model_names
from lib.stt import init_stt, stt, stt_model_names
from lib.tts import init_tts, tts, tts_model_names
from lib.llm import init_llm, llm, llm_model_names


def set_quick_edit_mode(turn_on=None) -> bool:
    """Enable/Disable windows console Quick Edit Mode"""
    import win32console  # pyright: ignore[reportMissingModuleSource]

    ENABLE_QUICK_EDIT_MODE = 0x40
    ENABLE_EXTENDED_FLAGS = 0x80

    screen_buffer = win32console.GetStdHandle(-10)
    orig_mode = screen_buffer.GetConsoleMode()
    is_on = orig_mode & ENABLE_QUICK_EDIT_MODE
    if is_on != turn_on and turn_on is not None:
        if turn_on:
            new_mode = orig_mode | ENABLE_QUICK_EDIT_MODE
        else:
            new_mode = orig_mode & ~ENABLE_QUICK_EDIT_MODE
        screen_buffer.SetConsoleMode(new_mode | ENABLE_EXTENDED_FLAGS)

    return is_on if turn_on is None else turn_on


if os.name == "nt" and sys.stdout.isatty():
    set_quick_edit_mode(False)


class Config(TypedDict):
    tts_model_name: str
    stt_model_name: str
    llm_model_name: str
    embed_model_name: str
    use_disk_cache: bool
    host: str
    port: int


def load_config() -> Config:
    config: dict | Config = {}
    try:
        with open("aiserver.config.json", "r") as f:
            config = Config(**json.load(f))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass

    if not "tts_model_name" in config:
        if os.environ.get("AISERVER_TTS_MODEL_NAME"):
            config["tts_model_name"] = os.environ["AISERVER_TTS_MODEL_NAME"]
        else:
            config["tts_model_name"] = pick(
                options=tts_model_names, title="Select a TTS model"
            )[0]

    if not "stt_model_name" in config:
        if os.environ.get("AISERVER_STT_MODEL_NAME"):
            config["stt_model_name"] = os.environ["AISERVER_STT_MODEL_NAME"]
        else:
            config["stt_model_name"] = pick(
                options=stt_model_names, title="Select a STT model"
            )[0]

    if not "llm_model_name" in config:
        if os.environ.get("AISERVER_LLM_MODEL_NAME"):
            config["llm_model_name"] = os.environ["AISERVER_LLM_MODEL_NAME"]
        else:
            config["llm_model_name"] = pick(
                options=llm_model_names, title="Select a LLM model"
            )[0]

    if not "use_disk_cache" in config:
        if os.environ.get("AISERVER_USE_DISK_CACHE"):
            config["use_disk_cache"] = (
                os.environ["AISERVER_USE_DISK_CACHE"].lower() == "true"
            )
        else:
            config["use_disk_cache"] = (
                pick(
                    ["Disabled", "Enabled"],
                    "Enable LLM Disk cache? This may speed up response times if the disk is faster than prompt evaluation, but also doubles memory usage.",
                )[0]
                == "Enabled"
            )

    if not "embed_model_name" in config:
        if os.environ.get("AISERVER_EMBED_MODEL_NAME"):
            config["embed_model_name"] = os.environ["AISERVER_EMBED_MODEL_NAME"]
        else:
            config["embed_model_name"] = pick(
                options=embed_model_names, title="Select an Embedding model"
            )[0]
        # config["embed_model_name"] = "None"

    if not "host" in config:
        if os.environ.get("AISERVER_HOST"):
            config["host"] = os.environ["AISERVER_HOST"]
        else:
            config["host"] = (
                input("Enter the IP to bind or leave empty for default [127.0.0.1]: ")
                or "127.0.0.1"
            )

    if not "port" in config:
        if os.environ.get("AISERVER_PORT"):
            config["port"] = int(os.environ["AISERVER_PORT"])
        else:
            config["port"] = int(
                input("Enter the port number or leave empty for default port [8080]: ")
                or "8080"
            )

    if config["port"] < 1025 or config["port"] > 65535:
        raise ValueError("Port number must be between 1025 and 65535")

    print(f'Selected STT model: {config["stt_model_name"]}')
    print(f'Selected LLM model: {config["llm_model_name"]}')
    print(f'Selected TTS model: {config["tts_model_name"]}')
    print(f'Selected Embedding model: {config["embed_model_name"]}')
    return config


config = load_config()

llm_model = init_llm(config["llm_model_name"], config["use_disk_cache"])
tts_model = init_tts(config["tts_model_name"])
stt_model = init_stt(config["stt_model_name"])
embed_model = init_embed(config["embed_model_name"])


# Create the FastAPI application
app = FastAPI()


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def create_chat_completion(chat: dict):
    """
    Create a chat completion using the LLM.
    """
    if not llm_model:
        return JSONResponse(content={"error": "model not found"}, status_code=400)

    if "messages" not in chat:
        return JSONResponse(content={"error": "messages is required"}, status_code=400)

    # Execute the LLM call
    completion = llm(llm_model, chat)
    return completion


@app.post("/v1/audio/speech")
@app.post("/audio/speech")
async def create_speech(request: Request):
    """
    Given text input (and optional speed/voice/format), generate TTS output.
    This endpoint streams audio using an async generator.
    """
    data = await request.json()
    text_input = data.get("input", "")
    speed = data.get("speed", 1.0)
    voice = data.get("voice", "nova")
    response_format = data.get("response_format", "wav")

    stream = await tts(tts_model, text_input, speed, voice, response_format)

    # Return a streaming response
    return StreamingResponse(stream, media_type=f"audio/{response_format}")


@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def create_transcription(
    language: str = Form("en"), file: UploadFile = File(...)  # default language is 'en'
):
    """
    Transcribe an audio file using the STT model.
    """
    contents = await file.read()
    text, info = stt(stt_model, contents, language)

    # Return text in a JSON payload
    return {"text": "\n".join(text)}


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def create_embedding(data: dict):
    """
    Create embeddings from input text using the embedding model.
    """
    if not data or "input" not in data:
        return JSONResponse(content={"error": "input is required"}, status_code=400)

    embedding = embed(embed_model, data)
    return embedding


def main():
    uvicorn.run(app, host=config["host"], port=config["port"], log_level="info")


"""
sample curl request to create a speech:
curl -X POST "http://localhost:8080/v1/audio/speech" -H "Content-Type: application/json" -d '{"input":"Hello World.", "response_format":"raw", "voice":"nova"}' | aplay -r 24000 -f S16_LE
curl -X POST "http://localhost:8080/v1/audio/speech" -H "Content-Type: application/json" -d '{"input":"Hello World.", "response_format":"wav", "voice":"nova"}' > audio.wav

sample curl request to create a transcription:
curl -X POST "http://localhost:8080/v1/audio/transcriptions" -F "file=@./audio.wav" -F "language=en"

sample curl request to create a chat completion:
curl -X POST "http://localhost:8080/v1/chat/completions" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello, how are you?"}]}' | jq

sample curl request to create an embedding:
curl -X POST "http://localhost:8080/v1/embeddings" -H "Content-Type: application/json" -d '{"input":"Hello, world!"}' | jq
"""

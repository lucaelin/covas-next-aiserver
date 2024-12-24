import json
import re
import tempfile
import requests

default_config = {
    "host": "127.0.0.1",
    "port": 8080,
    "tts_model_name": "vits-piper-en_US-ljspeech-high.tar.bz2",
    "stt_model_name": "tiny.en",
    "llm_model_name": "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
    "use_disk_cache": False,
}


def test_server_executable():
    import subprocess
    import os

    # create temp dir
    temp_dir = tempfile.mkdtemp()

    # write config.json to temp dir
    with open(f"{temp_dir}/aiserver.config.json", "w") as f:
        f.write(json.dumps(default_config))

    # run ../../dist/Chat/Chat.exe relative to this file, with temp dir as working directory
    server_location = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../dist/AIServer/AIServer.exe"
    )
    proc = subprocess.Popen(
        [server_location],
        cwd=temp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        shell=False,
        close_fds=True,
    )

    # read stdout until server outputs "Running"
    while proc.stdout:
        line = proc.stdout.readline()
        if not line:
            break
        print(line)
        if "Running on http://127.0.0.1:8080" in line:
            break

    # test http api
    # /chat/completions
    response = requests.post(
        "http://127.0.0.1:8080/chat/completions",
        json={
            "model": "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
            "prompt": {
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
                "max_tokens": 1,
            },
        },
    )
    assert response.status_code == 200
    assert "choices" in response.json()
    assert len(response.json()["choices"]) > 0
    assert "message" in response.json()["choices"][0]
    assert "content" in response.json()["choices"][0]["message"]
    assert len(response.json()["choices"][0]["message"]["content"]) > 0

    # /audio/speech
    response = requests.post(
        "http://127.0.0.1:8080/audio/speech",
        json={"input": "Hello world.", "voice": "nova", "speed": 1.0},
    )
    assert response.status_code == 200
    # save audio file to temp_dir
    with open(f"{temp_dir}/audio.wav", "wb") as f:
        f.write(response.content)

    # /audio/transcriptions
    response = requests.post(
        "http://127.0.0.1:8080/audio/transcriptions",
        files={"file": open(f"{temp_dir}/audio.wav", "rb")},
    )
    assert response.status_code == 200
    transcription = response.json()
    assert "text" in transcription
    assert transcription["text"] == "Hello world."

    # terminate Chat.exe
    proc.kill()
    proc.wait()


if __name__ == "__main__":
    test_server_executable()

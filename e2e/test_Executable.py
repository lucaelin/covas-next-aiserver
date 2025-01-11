import json
import tempfile
from time import sleep
import requests
from openai import OpenAI

default_config = {
    "host": "127.0.0.1",
    "port": 8080,
    "tts_model_name": "vits-piper-en_US-ljspeech-high.tar.bz2",
    "stt_model_name": "tiny.en",
    "llm_model_name": "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
    "embed_model_name": "lmstudio-community/granite-embedding-107m-multilingual-GGUF",
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
        os.path.dirname(os.path.abspath(__file__)), "../dist/AIServer/AIServer"
    )
    proc = subprocess.Popen(
        [server_location],
        cwd=temp_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        encoding="utf-8",
        errors="ignore",
        shell=False,
        close_fds=True,
    )

    # read stdout until server outputs "Running"
    while proc.stdout:
        line = proc.stdout.readline()
        if not line:
            break
        print(line)
        if "running on http://127.0.0.1:8080" in line:
            break

    # read and print the rest of the output in a thread
    import threading

    def read_output(pipe):
        while pipe:
            line = pipe.readline()
            if not line:
                break
            print(line)

    t = threading.Thread(target=read_output, args=(proc.stdout,))
    t.start()
    t2 = threading.Thread(target=read_output, args=(proc.stderr,))
    t2.start()

    openai = OpenAI(base_url="http://127.0.0.1:8080", api_key="-")

    # test http api
    # /chat/completions
    response = None
    while not response:
        print("Testing /chat/completions")
        sleep(3)
        try:
            response = openai.chat.completions.create(
                model="lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=1,
            )
            print(response)
            print(response.choices[0].message.content)
        except requests.exceptions.ConnectionError as e:
            print(e)

    assert isinstance(response.choices[0].message.content, str)
    assert len(response.choices[0].message.content) > 0

    # /audio/speech
    print("Testing /audio/speech")
    response = openai.audio.speech.create(
        model="-", input="Hello world.", voice="nova", speed=1.0, response_format="wav"
    )

    assert isinstance(response.content, bytes)
    assert len(response.content) > 0
    assert response.content[:4] == b"RIFF"
    with open(f"{temp_dir}/audio.wav", "wb") as f:
        f.write(response.content)

    # /audio/transcriptions
    print("Testing /audio/transcriptions")
    response = openai.audio.transcriptions.create(
        model="-", file=open(f"{temp_dir}/audio.wav", "rb")
    )
    assert isinstance(response.text, str)
    assert len(response.text) > 0

    response = openai.embeddings.create(model="-", input="Hello, world!")
    assert isinstance(response.data[0].embedding, list)
    assert len(response.data[0].embedding) > 0
    assert all(isinstance(x, float) for x in response.data[0].embedding)

    # terminate Chat.exe
    proc.kill()
    proc.wait()


if __name__ == "__main__":
    test_server_executable()

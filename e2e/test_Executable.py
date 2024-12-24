import json
import tempfile

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

    # read stdout until Chat.exe either exits or outputs "System Ready."
    while proc.stdout:
        line = proc.stdout.readline()
        if not line:
            break
        print(line)
        if "Running on http://127.0.0.1:8080" in line:
            break

    # assert that Chat.exe is running
    assert proc.poll() is None

    # terminate Chat.exe
    proc.kill()
    proc.wait()


if __name__ == "__main__":
    test_server_executable()

#!/bin/bash

apt-get update
apt-get install -y python3 python3-pip python3-venv cmake build-essential
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
#CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --upgrade --no-deps --no-cache-dir llama-cpp-python==0.2.90
pip install llama-cpp-python==0.2.90

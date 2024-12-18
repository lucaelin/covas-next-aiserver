#!/bin/bash

apt-get update && apt-get install -y cmake libcurl4-gnutls-dev nano
git clone https://github.com/ggerganov/llama.cpp
cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=ON
cmake --build llama.cpp/build --config Release -j8
cp llama.cpp/build/bin/llama-quantize llama.cpp/
git clone https://github.com/lucaelin/covas-next-aiserver
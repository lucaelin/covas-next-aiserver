#!/bin/bash

#python -c 'import os, llama_cpp; print(os.path.join(os.path.dirname(llama_cpp.__file__), "lib", "libonnxruntime_providers_shared.so"))'
export ONNXRUNTIME_DLL="/home/luca/.cache/pypoetry/virtualenvs/covas-next-aiserver-bKi0qKEh-py3.10/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so"
#python -c 'import os, llama_cpp; print(os.path.join(os.path.dirname(llama_cpp.__file__), "lib", "libllama.so"))'
export LLAMACPP_DLL="/home/luca/.cache/pypoetry/virtualenvs/covas-next-aiserver-bKi0qKEh-py3.10/lib/python3.10/site-packages/llama_cpp/lib/libllama.so"

echo $ONNXRUNTIME_DLL
echo $LLAMACPP_DLL
pyinstaller ./src/AIServer.py -y --onedir --clean --console --hidden-import=comtypes.stream --add-binary $ONNXRUNTIME_DLL:. --add-binary $LLAMACPP_DLL:./llama_cpp/lib
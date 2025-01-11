#!/bin/bash

#python -c 'import os, llama_cpp; print(os.path.join(os.path.dirname(llama_cpp.__file__), "lib", "libonnxruntime_providers_shared.so"))'
export ONNXRUNTIME_DLL="/home/luca/.pyenv/versions/3.10.14/envs/aiserver/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so"
#export ONNXRUNTIME_DLL=$(python -c 'import os, onnxruntime; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi", "libonnxruntime_providers_shared.so"))')
#python -c 'import os, llama_cpp; print(os.path.join(os.path.dirname(llama_cpp.__file__), "lib", "libllama.so"))'
export LLAMACPP_DLL="/home/luca/.pyenv/versions/3.10.14/envs/aiserver/lib/python3.10/site-packages/llama_cpp/lib/libllama.so"
#export LLAMACPP_DLL=$(python -c 'import os, llama_cpp; print(os.path.join(os.path.dirname(llama_cpp.__file__), "lib", "libllama.so"))')

echo $ONNXRUNTIME_DLL
echo $LLAMACPP_DLL
pyinstaller ./src/AIServer.py -y --onedir --clean --console --hidden-import=comtypes.stream --collect-all language_tags --add-binary $ONNXRUNTIME_DLL:. --add-binary $LLAMACPP_DLL:./llama_cpp/lib
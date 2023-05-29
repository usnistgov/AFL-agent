#!/bin/bash -i

conda activate afl_agent
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64/

if [[ -z "${TILED_API_KEY}" ]]; then
  export TILED_API_KEY=$(cat ~/.afl/tiled_api_key)
else
  export TILED_API_KEY="${TILED_API_KEY}"
fi

if [[ -z "${AFL_SYSTEM_SERIAL}" ]]; then
  export AFL_SYSTEM_SERIAL=$(cat ~/.afl/system_serial)
else
  export AFL_SYSTEM_SERIAL="${AFL_SYSTEM_SERIAL}"
fi

python ~/AFL-agent/server_scripts/Multimodal_Agent.py

#!/bin/bash -i

git config --global credential.helper store

conda activate afl_agent

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


cd ~/AFL-agent/
git pull

python ~/AFL-agent/server_scripts/SampleServer_AL_SAS.py

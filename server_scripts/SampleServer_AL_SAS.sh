#!/bin/bash -i

git config --global credential.helper store

conda activate afl_agent

cd ~/AFL-agent/
git pull

python ~/AFL-agent/server_scripts/SampleServer_AL_SAS.py

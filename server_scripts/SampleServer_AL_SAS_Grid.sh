#!/bin/bash -i

git config --global credential.helper store

conda activate afl_agent

cd ~/afl642/
git pull

python ~/AFL-agent/server_scripts/SampleServer_AL_SAS_Grid.py

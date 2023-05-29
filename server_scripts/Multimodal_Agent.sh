#!/bin/bash -i

conda activate afl_agent
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64/

python ~/AFL-agent/server_scripts/Multimodal_Agent.py

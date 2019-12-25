#!/usr/bin/env bash
cd ./code
mkdir -p attention
cd -
python3.5 ./code/attention.py --testdata_dir ${1}\
                  --model_dir ${2}
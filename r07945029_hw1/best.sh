#!/usr/bin/env bash
cd ./code
mkdir -p best
cd -
python3.5 ./code/best.py --testdata_dir ${1}\
                  --model_dir ${2}
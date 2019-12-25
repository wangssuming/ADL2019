#!/usr/bin/env bash
cd ./code
mkdir -p rnn
cd -
python ./code/rnn.py --testdata_dir ${1}\
                  --model_dir ${2}
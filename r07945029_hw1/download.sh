#!/usr/bin/env bash
cd ./code
mkdir -p data
cd -
curl -o ./code/data/config_rnn.json https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/config_rnn.json
curl -o ./code/data/config_attention.json https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/config_attention.json
curl -o ./code/data/config_best.json https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/config_best.json
curl -o ./code/data/test.json https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/test.json
curl -o ./code/data/valid.json https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/valid.json
curl -o ./code/data/embedding.pkl https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/embedding.pkl
curl -o ./code/data/rnn.pkl https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/rnn.pkl
curl -o ./code/data/attention.pkl https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/attention.pkl
curl -o ./code/data/best.pkl https://raw.githubusercontent.com/wangssuming/adlhw1_model/master/best.pkl
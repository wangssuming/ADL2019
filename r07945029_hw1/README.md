### All commands tested under terminal for ubuntu on twgc-0.5A and git bash for windows

# ADL2019/hw1
Dialogue Modeling
* [Homework 1 Website](https://www.csie.ntu.edu.tw/~miulab/s107-adl/A1)
* [Homework 1 Slide](https://docs.google.com/presentation/d/15LCy7TkJXl2pdz394gSPKY-fkwA3wFusv7h01lWOMSw/edit#slide=id.p)
* [Kaggle Competition](https://www.kaggle.com/c/adl2019-homework-1)
    * Public Leaderboard Rank: 1/99
    * Private Leaderboard Rank: 2/99
* [Example Code](https://drive.google.com/file/d/1KLOEg7x64BAIk8nFJwXaf9eczGjV667e/view)
* [Data](https://www.kaggle.com/c/13262/download-all)


## 0. Requirements
```
torch==1.0.1
tqdm==4.28.1
nltk==3.4
numpy==1.15.4
```

## 1. Data Preprocessing

### 1. Prepare the dataset and pre-trained embeddings (FastText is used here) in `./data`.
```
./data/train.json
./data/valid.json
./data/test.json
./data/crawl-300d-2M.vec
```


## 1. Train

### 1. Preprocess the data
```
cd ./ADL2019/hw1/src
python make_dataset.py ../data/
```

### 2. Training and Prediction
```
python train.py ../models/
python predict.py ../models/ --epoch -1
```


## 2. Predict

### 1. Download all data: 
```
bash download.sh

If error curl: (3) Illegal characters found in URL appear, delete first line(#!/usr/bin/env bash) in download.sh and run again.

The download data will be under ./code/data/
```

### 2. Run rnn prediction: (python 3.5 or3.6, specify 3.6 in .sh)
```
bash rnn.sh ./code/data/test.json ./code/rnn/predict.csv

If error : invalid optionne 4: cd: - appear, delete first line(#!/usr/bin/env bash) in rnn.sh and run again.

The prediction csv file will be under ./code/rnn/
```

### 3. Run attention prediction: (python 3.5 ro 3.6, specify 3.6 in .sh)
```
bash attention.sh ./code/data/test.json ./code/attention/predict.csv

If error : invalid optionne 4: cd: - appear, delete first line(#!/usr/bin/env bash) in attention.sh and run again.

The prediction csv file will be under ./code/attention/
```

### 4. Run best prediction: (python 3.5 ro 3.6, specify 3.6 in .sh)
```
bash best.sh ./code/data/test.json ./code/best/predict.csv

If error : invalid optionne 4: cd: - appear, delete first line(#!/usr/bin/env bash) in best.sh and run again.

The prediction csv file will be under ./code/best/
```


## 3. Run attention weight visualization: (python 3.6, test under git bash for windows)

```
cd ./code
python visualization.py

matplotlib package is needed.
```


###### tags: `NTU` `ADL` `2019`
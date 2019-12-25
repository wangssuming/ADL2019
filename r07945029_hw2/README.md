## Requirements
1. Python=3.6

2. Install the following required packages.
    ```
    pip install overrides
    pip install h5py
    ```

## Part1
cd part1

1. dowmload data
    ```
    bash download.sh
    ```

2. predict
    ```
    bash simple.sh
    ```
    
3. prediction file will be under model/submission folder.


## Part2
cd part2

*need to run strong.sh first, then best.sh

1. requirements.txt
    if requirements.txt not work:
    ```
    python3.6 -m pip install flair
    python3.6 -m pip install allennlp
    ```

2. Create dataset object from raw data.
    ```
    bash download.sh
    ```

3. predict strong
    ```
    sudo su
    bash ./strong.sh ./code/data/classification/test.csv ./strong/predict.csv
    ```
 
4. predict strong
    ```
    sudo su
    bash ./strong.sh ./code/data/classification/test.csv ./strong/predict.csv
    ```
    
5. prediction file will be under strong folder and test folder separately.

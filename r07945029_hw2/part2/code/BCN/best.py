import argparse
import csv
import random
import sys
from pathlib import Path
import pickle
import re

import ipdb
import spacy
from box import Box
from tqdm import tqdm

from dataset import Part1Dataset
import numpy as np
import torch

import os
import numpy as np
import pandas as pd
from glob import glob

from dataset import create_data_loader
from train import Model
from common.utils import load_pkl
from ELMo.embedder import Embedder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='Data directory')
    parser.add_argument('--save_model_dir', dest='save_model_dir', type=str, help='Model directory')
    args = parser.parse_args()

    return vars(args)


def load_data(mode, data_path, nlp):
    print('[*] Loading {} data from {}'.format(mode, data_path))
    with data_path.open(encoding="utf8") as f:
        reader = csv.DictReader(f)
        data = [r for r in reader]

    for d in tqdm(data, desc='[*] Tokenizing', dynamic_ncols=True):
        text = re.sub('-+', ' ', d['text'])
        text = re.sub('\s+', ' ', text)
        doc = nlp(text)
        d['text'] = [token.text for token in doc]
    print('[-] {} data loaded\n'.format(mode.capitalize()))

    return data


def create_dataset(data, word_vocab, char_vocab, dataset_dir):
    for m, d in data.items():
        print('[*] Creating {} dataset'.format(m))
        dataset = Part1Dataset(d, word_vocab, char_vocab)
        dataset_path = (dataset_dir / '{}.pkl'.format(m))
        with dataset_path.open(mode='wb') as f:
            pickle.dump(dataset, f)
        print('[-] {} dataset saved to {}\n'.format(m.capitalize(), dataset_path))


def test(data_dir, dataset_dir, word_vocab, char_vocab):
    try:
#        print(dataset_dir / 'config.yaml')
        cfg = Box.from_yaml(filename=dataset_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Dataset directory({}) must contain config.yaml'.format(dataset_dir))
        exit(1)
    print('[-] Vocabs and datasets will be saved to {}\n'.format(dataset_dir))

#    output_files = ['word.pkl', 'char.pkl', 'test.pkl']
#    if any([(dataset_dir+p).exists() for p in output_files]):
#        print('[!] Directory already contains saved vocab/dataset')
#        exit(1)

    nlp = spacy.load('en')
    nlp.disable_pipes(*nlp.pipe_names)

    data_dir = Path(cfg.data_dir)
    data = {m: load_data(m, data_dir / '{}.csv'.format(m), nlp)
            for m in ['test']}
    create_dataset(data, word_vocab, char_vocab, dataset_dir)



def main(data_dir, save_model_dir):
    model_dir = Path('./code/data/model/MODEL_NAME')
    prediction_dir = model_dir / 'predictions'
            
    p1 = model_dir / 'predictions/predict-1.csv'
    p2 = model_dir / 'predictions/predict-2.csv'
    p3 = model_dir / 'predictions/predict-3.csv'
    p4 = model_dir / 'predictions/predict-4.csv'
    p5 = model_dir / 'predictions/predict-5.csv'
    p6 = model_dir / 'predictions/predict-6.csv'
    p7 = model_dir / 'predictions/predict-7.csv'
    if any([p.exists() for p in [p1, p2, p3, p4, p5, p6, p7]]):
        ensemble(save_model_dir)
    else:
        if not prediction_dir.exists():
            prediction_dir.mkdir()
            print('[-] Directory {} created'.format(prediction_dir))
        model_path(data_dir, save_model_dir, model_dir, prediction_dir)


def model_path(data_dir, save_model_dir, model_dir, prediction_dir):
    dataset_dir = Path('./code/data/classification')
    batch_size = 4
    word_vocab = load_pkl(dataset_dir / 'word.pkl')
    char_vocab = load_pkl(dataset_dir / 'char.pkl')
    test(data_dir, dataset_dir, word_vocab, char_vocab)
    try:
        cfg1 = Box.from_yaml(filename=model_dir / 'config1.yaml')
        cfg2 = Box.from_yaml(filename=model_dir / 'config2.yaml')
        cfg3 = Box.from_yaml(filename=model_dir / 'config3.yaml')
        cfg4 = Box.from_yaml(filename=model_dir / 'config4.yaml')
        cfg5 = Box.from_yaml(filename=model_dir / 'config5.yaml')
        cfg6 = Box.from_yaml(filename=model_dir / 'config6.yaml')
        cfg7 = Box.from_yaml(filename=model_dir / 'config7.yaml')
    except FileNotFoundError:
        print('[!] Model directory({}) must contain config.yaml'.format(model_dir))
        exit(1)
        
#    dataset_dir = Path('./dataset/classification')
    test_dataset_path = dataset_dir / 'test.pkl'
    ckpt_path1 = model_dir / 'ckpts' / 'predict1.ckpt'
    ckpt_path2 = model_dir / 'ckpts' / 'predict2.ckpt'
    ckpt_path3 = model_dir / 'ckpts' / 'predict3.ckpt'
    ckpt_path4 = model_dir / 'ckpts' / 'predict4.ckpt'
    ckpt_path5 = model_dir / 'ckpts' / 'predict5.ckpt'
    ckpt_path6 = model_dir / 'ckpts' / 'predict6.ckpt'
    ckpt_path7 = model_dir / 'ckpts' / 'predict7.ckpt'
    print('[-] Test dataset: {}'.format(test_dataset_path))
#    print('[-] Model checkpoint: {}\n'.format(ckpt_path))

    print('[*] Loading vocabs and test dataset from {}'.format(dataset_dir))
    test_dataset = load_pkl(test_dataset_path)
    
    model_predict(cfg1, ckpt_path1, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size)
    model_predict(cfg2, ckpt_path2, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size)
    model_predict(cfg3, ckpt_path3, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size)
    model_predict(cfg4, ckpt_path4, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size)
    model_predict(cfg5, ckpt_path5, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size)
    model_predict(cfg6, ckpt_path6, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size)
    model_predict(cfg7, ckpt_path7, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size)
    ensemble(save_model_dir)


def ensemble(save_model_dir):
    fnames = glob('./code/data/model/MODEL_NAME/predictions/*.csv')
    print(fnames)
    print(len(fnames))
    NUM_OPTIONS = 5
    num_files = len(fnames)
    num_rows = pd.read_csv(fnames[0])['Id'].values.shape[0]
    preds = np.zeros((num_files, num_rows, NUM_OPTIONS)).astype(np.uint8)
    for f, fname in enumerate(fnames):
        Predict = pd.read_csv(fname)['label'].values.astype(int)
        for r, label in enumerate(Predict):
            preds[f, r, label - 1] = 1
    sum_preds = np.sum(preds, axis=0)
    print (sum_preds.shape)
    best_ids = np.argmax(sum_preds, axis=-1) + 1
    submit = pd.read_csv(fnames[0])
    ensembled_labels = []
    for r, name in enumerate(submit['Id']):
        ensembled_labels.append(best_ids[r])
    submit['label'] = ensembled_labels
    submit.to_csv(Path(save_model_dir), index=False)



def model_predict(cfg, ckpt_path, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size):
    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print('[*] Creating test data loader\n')
    if batch_size:
        cfg.data_loader.batch_size = batch_size
    data_loader = create_data_loader(
        test_dataset, word_vocab, char_vocab, **cfg.data_loader, shuffle=False)

    if cfg.use_elmo:
        print('[*] Creating ELMo embedder')
        elmo_embedder = Embedder(**cfg.elmo_embedder)
    else:
        elmo_embedder = None

    print('[*] Creating model\n')
    cfg.net.n_ctx_embs = cfg.elmo_embedder.n_ctx_embs if cfg.use_elmo else 0
    cfg.net.ctx_emb_dim = cfg.elmo_embedder.ctx_emb_dim if cfg.use_elmo else 0
    model = Model(device, word_vocab, char_vocab, cfg.net, cfg.optim)

    model.load_state(ckpt_path)

    Ids, predictions = predict(
        device, data_loader, cfg.data_loader.max_sent_len, elmo_embedder, model)
    print("predict_done")
    save_predictions(Ids, predictions, prediction_dir / 'predict-{}.csv'.format(cfg.elmo_embedder.predict))


def predict(device, data_loader, max_sent_len, elmo_embedder, model):
    model.set_eval()
    with torch.no_grad():
        Ids = []
        predictions = []
        bar = tqdm(data_loader, desc='[Predict]', leave=False, dynamic_ncols=True, ascii=True)
        for batch in bar:
            Ids += batch['Id']
            text_word = batch['text_word'].to(device=device)
            text_char = batch['text_char'].to(device=device)
            if elmo_embedder and elmo_embedder.ctx_emb_dim > 0:
                text_ctx_emb = elmo_embedder(batch['text_orig'], max_sent_len)
                text_ctx_emb = torch.tensor(text_ctx_emb, device=device)
            else:
                text_ctx_emb = torch.empty(
                    (*text_word.shape, 0), dtype=torch.float32, device=device)
            text_pad_mask = batch['text_pad_mask'].to(device=device)
            logits = model(text_word, text_char, text_ctx_emb, text_pad_mask)
            label = logits.max(dim=1)[1]
            predictions += label.tolist()
        bar.close()

    return Ids, predictions


def save_predictions(Ids, predictions, output_path):
    with output_path.open(mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'label'])
        writer.writeheader()
        writer.writerows(
            [{'Id': Id, 'label': p + 1} for Id, p in zip(Ids, predictions)])
    print('[-] Output saved to {}'.format(output_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)

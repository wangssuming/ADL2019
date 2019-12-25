# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:39:41 2019

@author: hb2506
"""

import pdb
import pickle
import sys
import traceback
import json
import torch
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from preprocessor1 import Preprocessor
import dataset
import torch.nn.functional as F



def forward(context, context_lens, option, option_lens):
    dim_embeddings = 300
    hidden = 128
    rnn1 = torch.nn.GRU(dim_embeddings, hidden, 1, bidirectional=True, batch_first=True)
    nn = torch.nn.Linear(hidden*2, hidden*2, bias=False)
    attention = torch.nn.Linear(hidden*2, 1, bias=False)
    context = torch.unsqueeze(context, 0)
    option = torch.unsqueeze(option, 0)
    context_out, context_hidden = rnn1(context)
    context_nn = nn(context_out)
    option_out, option_hidden = rnn1(option)
    option_nn = nn(option_out) 
    C = torch.unsqueeze(context_nn, 1).expand(-1, option.size(1), -1, -1)
    P = torch.unsqueeze(option_nn, 2).expand(-1, -1, context.size(1), -1)
    attention = attention(torch.tanh(P + C)).squeeze(dim=-1)
    attention = torch.squeeze(attention, 0)
    atten_softmax_con = F.softmax(attention,dim=0)
    atten_softmax_op = F.softmax(attention,dim=1)
    return attention, atten_softmax_con, atten_softmax_op 
     
def showAttention(input_sentence, output_words, attentions, attentions_con, attentions_op):
    fig = plt.figure()
    fig.suptitle("Attention weight plot", fontsize=20)
    ax = fig.add_subplot(311)
    cax1 = ax.matshow(attentions, cmap='PuBu')
    fig.colorbar(cax1)
    # Set up axes
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)   
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Attention without softmax", fontsize=17)
    
    ax = fig.add_subplot(312)
    cax2 = ax.matshow(attentions_op, cmap='YlGnBu')
    fig.colorbar(cax2)
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Attention with softmax on context dim", fontsize=17)
    
    ax = fig.add_subplot(313)
    cax3 = ax.matshow(attentions_con, cmap='GnBu')
    fig.colorbar(cax3)
    ax.set_xticklabels([''] + input_sentence, rotation=90)
    ax.set_yticklabels([''] + output_words)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Attention with softmax on option dim", fontsize=17)

    plt.show()


def main():
    
    # load embedding
    with open("data/embedding.pkl", 'rb') as f:
        embedding = pickle.load(f)
    
    preprocessor = Preprocessor(None)
    
    # update embedding used by preprocessor
    preprocessor.embedding = embedding
    
    # predict test
    
    with open("data/valid.json") as f:
        validraw = json.load(f)
        
    input_sentence = ''
    output_sentence = ''
    for sample in validraw[0]["messages-so-far"]:
        input_sentence += sample["utterance"]
    for sample in validraw[0]["options-for-correct-answers"]:
        output_sentence += sample["utterance"] 

    attention_input, in_word = preprocessor.sentence_to_indices(input_sentence)
    attention_output, out_word = preprocessor.sentence_to_indices(output_sentence)
    padding = embedding.to_index('</s>')

    padded_len = min(300, len(attention_input))
    contexts = torch.tensor(dataset.pad_to_len(attention_input, padded_len, padding))
    padded_len = min(50, len(attention_output))
    options = torch.tensor(dataset.pad_to_len(attention_output, padded_len, padding))

    embedded = torch.nn.Embedding(embedding.vectors.size(0),embedding.vectors.size(1))
    embedded.weight = torch.nn.Parameter(embedding.vectors)
    contexts = embedded(contexts)
    options = embedded(options)
    
    attention, atten_softmax_con, atten_softmax_op = forward(contexts, contexts.size(), options, options.size())
    
    showAttention(in_word, out_word, attention.detach().numpy(), 
                  atten_softmax_con.detach().numpy(), atten_softmax_op.detach().numpy())
    
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)





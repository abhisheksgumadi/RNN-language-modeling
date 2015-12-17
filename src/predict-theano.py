# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:06:12 2015

@author: root
"""

import os
import sys
from rnn_theano import RNNTheano
from utils import load_model_parameters_theano

_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '100'))
_MODEL_FILE = (os.environ.get("/home/abhishek/deep_learning_tutorial/RNN/rnn-tutorial-rnnlm-master/2-layered.npz")) #os.environ.get('MODEL_FILE')

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
            print index_to_word[sampled_word]
            sys.stdout.flush()
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str
    
if __name__ == "__main__":
    
    model = RNNTheano(8000, hidden_dim=_HIDDEN_DIM)
    
    if _MODEL_FILE != None:
        print "loaded parameters"
        model = RNNTheano(word_dim=8000, hidden_dim=100)
        load_model_parameters_theano(_MODEL_FILE, model)
        
    num_sentences = 1
    senten_min_length = 3
     
    for i in range(num_sentences):
        print i
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence(model)
        print " ".join(sent)
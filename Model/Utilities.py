# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:44:01 2021

@author: kesha
"""

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import re
import glob


def tokenization(list_obj, num_words = None, oov_token = None, lower=False, filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n'):
    t = Tokenizer(oov_token=oov_token, lower=lower, filters=filters, num_words=num_words)
    t.fit_on_texts(list_obj)
    return t

def get_masked_input_and_labels(encoded_texts, mask_token_id):
    # 15% BERT masking
    inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
    # Do not mask special tokens
    inp_mask[encoded_texts <= 1] = False
    # Set targets to -1 by default, it means ignore
    labels = -1 * np.ones(encoded_texts.shape, dtype=int)
    # Set labels for masked tokens
    labels[inp_mask] = encoded_texts[inp_mask]

    # Prepare input
    encoded_texts_masked = np.copy(encoded_texts)
    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
    encoded_texts_masked[
        inp_mask_2mask
    ] = mask_token_id  # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
    encoded_texts_masked[inp_mask_2random] = np.random.randint(
        3, mask_token_id, inp_mask_2random.sum()
    )

    # Prepare sample_weights to pass to .fit() method
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    # y_labels would be same as encoded_texts i.e input tokens
    y_labels = np.copy(encoded_texts)

    return encoded_texts_masked, y_labels, sample_weights


class PreTrainGenerator(keras.utils.Sequence):
    'Batch Generator for Keras'
    def __init__(self, pretrain_universe, col_name, nrows, tokenizer_items, config, run_type = "model", batch_size=32, shuffle=True):        
        'Initialization'
        self.pretrain_universe = pretrain_universe
        self.col_name = col_name
        self.run_type = run_type
        self.nrows = nrows
        self.tokenizer_items = tokenizer_items
        self.mask_token_id = self.tokenizer_items.word_index["[mask]"]
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nrows / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch = self.pretrain_universe[index*self.batch_size:(index+1)*self.batch_size].index
        x_masked_items, y_masked_labels, sample_weights = self.data_preprocess(batch)
        X = x_masked_items
        Y = y_masked_labels
        Z = sample_weights
        return X, Y, Z

    def on_epoch_end(self):
        if self.shuffle:
            self.pretrain_universe = self.pretrain_universe.sample(frac=1)
        print("Data Generator: Epoch End")
    
    def data_preprocess(self, batch):
        'Processes data in batch_size samples'
        
        # Items
        pretrain_universe_sample = self.pretrain_universe.iloc[batch]
        txt_to_seq = self.tokenizer_items.texts_to_sequences(pretrain_universe_sample[self.col_name].tolist()) 
        padded_seq_items = pad_sequences(txt_to_seq, maxlen = self.config['MAX_LEN'], padding='pre')
      
        # Prepare data for pretrain model
        x_masked_items, y_masked_labels, sample_weights = get_masked_input_and_labels(
            padded_seq_items, self.mask_token_id
        )
        
        return x_masked_items, y_masked_labels, sample_weights


# !curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf aclImdb_v1.tar.gz

def get_text_list_from_files(files):
    text_list = []
    for name in files:
        with open(name, encoding='utf-8') as f: # encoding="utf-8", 'latin-1'
            for line in f:
                text_list.append(re.sub(r'[^\x00-\x7F]+',' ', line))
    return text_list


def get_data_from_text_files(folder_name):

    pos_files = glob.glob("aclImdb/" + folder_name + "/pos/*.txt")
    pos_texts = get_text_list_from_files(pos_files)
    neg_files = glob.glob("aclImdb/" + folder_name + "/neg/*.txt")
    neg_texts = get_text_list_from_files(neg_files)
    df = pd.DataFrame(
        {
            "review": pos_texts + neg_texts,
            "sentiment": [0] * len(pos_texts) + [1] * len(neg_texts),
        }
    )
    df = df.sample(len(df)).reset_index(drop=True)
    return df




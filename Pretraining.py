# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:44:01 2021

@author: kesha
"""

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pickle
import os
import argparse
from Model.Utilities import tokenization, PreTrainGenerator, get_data_from_text_files
from Model.Electra import encoder, Electra
from Configs.Pretraining_Config import config, raw_data_path, data_col_name, working_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_loc", help="Raw data location")
    parser.add_argument("--col_name", help="Text column to use for pretraining")
    parser.add_argument("--working_dir", help="Working directory to store model logs")
    parser.add_argument("--hparams", help="Model hyperparameters as dict")
    
    args = vars(parser.parse_args())
    if args['raw_data_loc'] is not None:
        raw_data_loc = args['raw_data_loc']
    else:
        raw_data_loc = raw_data_path
    if args['col_name'] is not None:
        col_name = args['col_name']
    else:
        col_name = data_col_name
    if args['working_dir'] is not None:
        wd = args['working_dir']
    else:
        wd = working_dir
    if args['hparams'] is not None:
        for key, val in args['hparams'].items():
            config[key] = val

# !curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf aclImdb_v1.tar.gz

# train_df = get_data_from_text_files("train")
# test_df = get_data_from_text_files("test")

# all_data = train_df.append(test_df)
# del train_df, test_df
# gc.collect()

# all_data.to_csv(config['RAW_DATA_PATH'], index=False)
all_data = pd.read_csv(raw_data_loc)

# Tokenize
tokenizer = tokenization(all_data[col_name], num_words=None, lower=True)
index_word = [tokenizer.index_word[i] for i in range(1, config['VOCAB_SIZE']-1)] + ['[mask]']
tokenizer = tokenization(index_word, num_words=None, lower=True)

print(len(tokenizer.word_index))
print(len(tokenizer.word_index))
mask_token_id = tokenizer.word_index["[mask]"]

# Pickle the config and weights
pickle.dump({'items': tokenizer}, open(wd+"tokenizer.pkl", "wb"))
config['VOCAB_SIZE'] = int(len(tokenizer.word_index))+1
pickle_byte_obj = [config]
pickle.dump(pickle_byte_obj, open(wd + "config.pkl", "wb"))
    

# Train and Save
train = PreTrainGenerator(all_data, col_name, len(all_data), tokenizer, config, run_type = "model", batch_size=config['PRETRAIN_BATCH_SIZE'], shuffle=True)    
train_steps = len(all_data)

generator_model, discriminator_model = encoder(max_len = config['MAX_LEN'], 
                                               vocab_size = config['VOCAB_SIZE'], 
                                               g_num_heads = config['G_NUM_HEAD'], 
                                               d_num_heads = config['D_NUM_HEAD'],                                                
                                               g_emb_dim = config['G_EMBED_DIM'], 
                                               d_emb_dim = config['D_EMBED_DIM'],
                                               shared_embedding = config['SHARED_EMBEDDINGS'],
                                               emb_type = config['EMBED_TYPE'],
                                               g_ff_dim = config['G_FF_DIM'], 
                                               d_ff_dim=config['D_FF_DIM'], 
                                               key_dim=config['ATTN_KEY_DIM'],
                                               g_num_layers = config['G_NUM_LAYERS'], 
                                               d_num_layers = config['D_NUM_LAYERS'],
                                               num_buckets = config['VOCAB_SIZE'] // config['COMPOSITION_BUCKET_SIZE'],
                                               g_layer_name_prefix = 'generator', 
                                               d_layer_name_prefix = 'discriminator')

print(generator_model.summary())
print(discriminator_model.summary())

electra_model = Electra(discriminator_model, generator_model, mask_token_id)
electra_model.compile(d_optimizer = keras.optimizers.Adam(learning_rate=config['D_LR'], beta_1=0.9, beta_2=0.999, epsilon=1e-6, clipnorm=1.0),
                      g_optimizer = keras.optimizers.Adam(learning_rate=config['G_LR'], beta_1=0.9, beta_2=0.999, epsilon=1e-6), 
                      d_loss_fn = keras.losses.BinaryCrossentropy(), # masked_sigmoid_cross_entropy
                      g_loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE))

electra_model.fit(train,
                  steps_per_epoch=int(train_steps//config['PRETRAIN_BATCH_SIZE']),
                  epochs=config['PRETRAIN_EPOCHS'])  
electra_model.save_weights(wd+"electra.h5")
discriminator_model.save(wd+"discriminator.h5")




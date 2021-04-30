# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 22:04:24 2021

@author: kesha
"""

raw_data_path = 'C:/Users/kesha/Desktop/Projects/Electra/Data/Data.csv'
data_col_name = 'review' # Text column to use for pretraining
working_dir = 'C:/Users/kesha/Desktop/Projects/Electra/Output/'

config = {
    'MAX_LEN': 256, # Sequence Length
    'VOCAB_SIZE': 30000, # Vocabulary Size
    'PRETRAIN_BATCH_SIZE': 8, # Batch Size - 8, 16, 32, 64, ...
    'PRETRAIN_EPOCHS': 5,
    'G_LR': 0.0005, # Generator learning rate
    'D_LR': 0.0005, # Discriminator learning rate
    'SHARED_EMBEDDINGS': True, # Boolean for shared embedding layers between discriminator and generator
    'G_EMBED_DIM': 64, # Generator embedding dimension (not used when shared embedding = False)
    'D_EMBED_DIM': 128, # Discriminator embedding dimension (used by generator when shared embedding = True)
    'ATTN_KEY_DIM': 64, # Multi-Headed attention key size
    'EMBED_TYPE': 'default', # default or compositional
    'COMPOSITION_BUCKET_SIZE': 5, # VOCAB_SIZE / COMPOSITION_BUCKET_SIZE (only used when EMBED_TYPE = compositional)
    'G_NUM_HEAD': 1, # Generator number of attention heads (recommended 1/4 of discriminator heads)
    'D_NUM_HEAD': 4, # Discriminator number of attention heads
    'G_FF_DIM': 256, # Generator FFN Size (recommended 1/4 of discriminator FFN)
    'D_FF_DIM': 768, # Discriminator FFN Size
    'G_NUM_LAYERS': 1, # Generator number of layers (recommended 1/4 of discriminator layers)
    'D_NUM_LAYERS': 4 # Discriminator number of layers
    }
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:44:01 2021

@author: kesha
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
import gc
from sklearn.model_selection import train_test_split
import os
import argparse
from Model.Electra import TransformerBlock, QREmbedding
from Model.Utilities import get_data_from_text_files
from Configs.Finetuning_Config import config, raw_data_path, working_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_loc", help="Raw data location")
    parser.add_argument("--working_dir", help="Working directory to store model logs")
    parser.add_argument("--hparams", help="Model hyperparameters as dict")
    
    args = vars(parser.parse_args())
    if args['raw_data_loc'] is not None:
        raw_data_loc = args['raw_data_loc']
    else:
        raw_data_loc = raw_data_path
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

# Get Tokenizer and Config
pickle_byte_obj = pickle.load(open(wd+"tokenizer.pkl", "rb"))
tokenizer = pickle_byte_obj['items']
# Get mask token id for masked language model
mask_token_id = tokenizer.word_index["[mask]"]

pretrain_config = pickle.load(open(wd+"config.pkl", "rb"))[0]


# Fine Tuning
discriminator = tf.keras.models.load_model(wd+"discriminator.h5", custom_objects = {'TransformerBlock': TransformerBlock, 'QREmbedding': QREmbedding})
# discriminator = discriminator_model
electra_pretrained_model = tf.keras.Model(
        discriminator.input, discriminator.get_layer("discriminator_{0}".format(pretrain_config['D_NUM_LAYERS']-1)).output
    )
electra_pretrained_model.trainable = False

for layer in electra_pretrained_model.layers:
    print(layer._name)
    print(layer.trainable)

def create_classifier_model():
    inputs = layers.Input((config['MAX_LEN'],), dtype=tf.int64)
    sequence_output = electra_pretrained_model(inputs)
    pooled_output = layers.GlobalMaxPooling1D()(sequence_output)
    hidden_layer = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.02))(pooled_output)
    hidden_layer = layers.LayerNormalization(epsilon=1e-6)(hidden_layer)
    outputs = layers.Dense(1, activation="sigmoid")(hidden_layer)
    classifer_model = keras.Model(inputs, outputs, name="classification")
    return classifer_model


classifer_model = create_classifier_model()
classifer_model.summary()


train, validation = train_test_split(all_data, test_size = 0.1, random_state = 0)
train.reset_index(drop=True, inplace=True)
validation.reset_index(drop=True, inplace=True)

txt_to_seq = tokenizer.texts_to_sequences(train['review'].tolist()) 
train_padded_seq = pad_sequences(txt_to_seq, maxlen = config['MAX_LEN'], padding='pre')

txt_to_seq = tokenizer.texts_to_sequences(validation['review'].tolist()) 
validation_padded_seq = pad_sequences(txt_to_seq, maxlen = config['MAX_LEN'], padding='pre')


# Train and Save
x_train = (
    tf.data.Dataset.from_tensor_slices((train_padded_seq, train.sentiment.values))
    .shuffle(1000)
    .batch(config['FINETUNE_BATCH_SIZE'])
)

x_validation = (
    tf.data.Dataset.from_tensor_slices((validation_padded_seq, validation.sentiment.values))
    .shuffle(1000)
    .batch(config['FINETUNE_BATCH_SIZE'])
)

train_steps = len(train)
validation_steps = len(validation)
del train, validation, train_padded_seq, validation_padded_seq
gc.collect()


classifer_model.compile(
optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

classifer_model.fit(
    x_train,
    epochs = config['FINETUNE_EPOCHS'],
    validation_data = x_validation,
)

# Unfreeze the weights
electra_pretrained_model.trainable = True
classifer_model.compile(
optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

classifer_model.fit(
    x_train,
    epochs = config['FINETUNE_EPOCHS'],
    validation_data = x_validation,
)

classifer_model.save(wd+"classifer_model.h5")
# val_accuracy: 0.8854 after 5 epochs 30k vocab size
# val_accuracy: 0.8914 after 5 epochs 100k vocab size

# classifer_model.built=True
# classifer_model.load_weights(wd+"classifer_model.h5")




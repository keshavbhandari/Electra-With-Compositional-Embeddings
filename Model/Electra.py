# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:44:01 2021

@author: kesha
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, emb_dim, ff_dim, key_dim, layer_name='TransformerBlock', dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(name=layer_name)
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.ff_dim = ff_dim
        self.key_dim = key_dim
        self.layer_name = layer_name
        self.dropout_rate = dropout_rate
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dropout_rate, name="encoder_{0}/multiheadattention".format(self.layer_name)) #self.emb_dim // self.num_heads
        self.ffn = keras.Sequential(
        [
            layers.Dense(self.ff_dim, activation="relu"),
            layers.Dense(self.emb_dim),
        ],
        name="{0}/ffn".format(self.layer_name))
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="{0}/att_layernormalization".format(self.layer_name))
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="{0}/ffn_layernormalization".format(self.layer_name))
        self.dropout1 = layers.Dropout(self.dropout_rate, name="{0}/att_dropout".format(self.layer_name))
        self.dropout2 = layers.Dropout(self.dropout_rate, name="{0}/ffn_dropout".format(self.layer_name))

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'emb_dim': self.emb_dim,
            'key_dim': self.key_dim,
            'ff_dim': self.ff_dim,
            'layer_name': self.layer_name,
            'dropout_rate': self.dropout_rate
        })
        return config

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


class QREmbedding(layers.Layer):
    def __init__(self, vocabulary, embedding_dim, num_buckets, name=None, **kwargs):
        super(QREmbedding, self).__init__(name=name)
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.num_buckets = num_buckets
        self.q_embeddings = layers.Embedding(num_buckets, embedding_dim,)
        self.r_embeddings = layers.Embedding(num_buckets, embedding_dim,)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'vocabulary': self.vocabulary,
            'embedding_dim': self.embedding_dim,
            'num_buckets': self.num_buckets
        })
        return config

    def call(self, inputs):
        # Get the item index.
        embedding_index = inputs
        # Get the quotient index.
        quotient_index = tf.math.floordiv(embedding_index, self.num_buckets)
        # Get the reminder index.
        remainder_index = tf.math.floormod(embedding_index, self.num_buckets)
        # Lookup the quotient_embedding using the quotient_index.
        quotient_embedding = self.q_embeddings(quotient_index)
        # Lookup the remainder_embedding using the remainder_index.
        remainder_embedding = self.r_embeddings(remainder_index)
        # Use multiplication as a combiner operation
        return quotient_embedding * remainder_embedding


def encoder(max_len = 256, vocab_size = 30000, g_num_heads = 1, d_num_heads = 1, 
            g_emb_dim = 64, d_emb_dim = 256, shared_embedding = False, 
            emb_type = "default", g_ff_dim = 128, d_ff_dim=768, 
            key_dim=64, g_num_layers = 1, d_num_layers = 1, num_buckets = None,
            g_layer_name_prefix = 'generator', d_layer_name_prefix = 'discriminator'):
        
    g_inputs = layers.Input((max_len,), dtype=tf.int64)
    d_inputs = layers.Input((max_len,), dtype=tf.int64)
    
    if shared_embedding:     
        g_emb_dim = d_emb_dim
        
        position_embeddings = layers.Embedding(
            input_dim=max_len,
            output_dim = d_emb_dim,
            weights=[get_pos_encoding_matrix(max_len, d_emb_dim)],
            name="position_embedding",
        )(tf.range(start=0, limit=max_len, delta=1))
        
        if emb_type == "compositional":
            if num_buckets is None:
                num_buckets = vocab_size // 5
            word_embeddings = QREmbedding(vocab_size, d_emb_dim, num_buckets = num_buckets, name="word_embedding")
        else:
            word_embeddings = layers.Embedding(vocab_size, d_emb_dim, name="word_embedding")
        g_word_embeddings = word_embeddings(g_inputs)
        d_word_embeddings = word_embeddings(d_inputs)
        
        g_embeddings = g_word_embeddings + position_embeddings
        d_embeddings = d_word_embeddings + position_embeddings
        
    else:
        g_position_embeddings = layers.Embedding(
            input_dim=max_len,
            output_dim = g_emb_dim,
            weights=[get_pos_encoding_matrix(max_len, g_emb_dim)],
            name="position_embedding",
        )(tf.range(start=0, limit=max_len, delta=1))
        
        d_position_embeddings = layers.Embedding(
            input_dim=max_len,
            output_dim = d_emb_dim,
            weights=[get_pos_encoding_matrix(max_len, d_emb_dim)],
            name="position_embedding",
        )(tf.range(start=0, limit=max_len, delta=1))
        
        if emb_type == "compositional":
            if num_buckets is None:
                num_buckets = vocab_size // 5
            g_word_embeddings = QREmbedding(vocab_size, g_emb_dim, num_buckets= num_buckets, name="g_word_embedding")
            d_word_embeddings = QREmbedding(vocab_size, d_emb_dim, num_buckets= num_buckets, name="d_word_embedding")
        else:
            g_word_embeddings = layers.Embedding(vocab_size, g_emb_dim, name="g_word_embedding")
            d_word_embeddings = layers.Embedding(vocab_size, d_emb_dim, name="d_word_embedding")
        g_word_embeddings = g_word_embeddings(g_inputs)
        d_word_embeddings = d_word_embeddings(d_inputs) 
        
        g_embeddings = g_word_embeddings + g_position_embeddings
        d_embeddings = d_word_embeddings + d_position_embeddings
    
    g_encoder_output = g_embeddings
    d_encoder_output = d_embeddings
    
    for i in range(g_num_layers):
        j = "{0}_{1}".format(g_layer_name_prefix, i)
        g_encoder_output = TransformerBlock(g_num_heads, g_emb_dim, g_ff_dim, key_dim, layer_name=j, dropout_rate=0.1)(g_encoder_output)
    
    for i in range(d_num_layers):
        j = "{0}_{1}".format(d_layer_name_prefix, i)
        d_encoder_output = TransformerBlock(d_num_heads, d_emb_dim, d_ff_dim, key_dim, layer_name=j, dropout_rate=0.1)(d_encoder_output)

    g_output = layers.Dense(vocab_size, name="{0}_output_1".format(g_layer_name_prefix), activation="softmax")(g_encoder_output)
    
    d_encoder_output = layers.GlobalMaxPooling1D()(d_encoder_output)
    d_output = layers.Dense(max_len, name="{0}_output_1".format(d_layer_name_prefix), activation="sigmoid")(d_encoder_output)
        
    g_model = keras.models.Model(inputs=g_inputs, outputs=g_output, name="{0}_model".format(g_layer_name_prefix))
    d_model = keras.models.Model(inputs=d_inputs, outputs=d_output, name="{0}_model".format(d_layer_name_prefix))

    return g_model, d_model


# Override train_step
class Electra(keras.Model):
    def __init__(self, discriminator, generator, mask_token_id):
        super(Electra, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.mask_token_id = mask_token_id

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(Electra, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.total_loss_metric = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def call(self, features, training=True):
        features = tf.cast(features, dtype = 'int64')
        predicted_tokens = self.generator(features, training=training)
        discriminator_tokens = tf.math.argmax(predicted_tokens, -1)
        tokens = tf.where(tf.equal(features, self.mask_token_id), discriminator_tokens, features)
        predictions = self.discriminator(tokens, training=training)
        return predicted_tokens, predictions

    def train_step(self, inputs):
        if len(inputs) == 3:
            features, labels, sample_weight = inputs
        else:
            features, labels = inputs
            sample_weight = None
        
        features = tf.cast(features, dtype = 'int64')
        labels = tf.cast(labels, dtype = 'int64')

        with tf.GradientTape(persistent=True) as tape:
            predicted_tokens, predictions = self(features, training=True)
            discriminator_tokens = tf.math.argmax(predicted_tokens, -1)
            original_or_replaced = tf.cast(tf.where(tf.equal(labels, discriminator_tokens), 0, 1), tf.float32)
            predictions = tf.cast(predictions, tf.float32)
            g_loss = self.g_loss_fn(labels, predicted_tokens, sample_weight=sample_weight)
            d_loss = self.d_loss_fn(original_or_replaced, predictions)
            total_loss = g_loss + 50 * d_loss
        
        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        
        d_grads = tape.gradient(total_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # Update metrics
        self.g_loss_metric.update_state(g_loss, sample_weight=sample_weight)
        self.d_loss_metric.update_state(d_loss)
        self.total_loss_metric.update_state(total_loss)
        return {
            "g_loss": self.g_loss_metric.result(),
            "d_loss": self.d_loss_metric.result(),
            "total_loss": self.total_loss_metric.result()
        }




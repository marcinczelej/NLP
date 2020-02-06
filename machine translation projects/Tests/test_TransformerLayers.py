import unittest
import sys
from parameterized import parameterized_class

import tensorflow as tf

sys.path.insert(0, r"../models")

from TransformerLayers import *
from TransformerModel import makePaddingMask, makeSequenceMask

@parameterized_class(("batch_size", "training_enabled"), [
        (1, True),
        (1, False),
        (16, True),
        (16, False),
    ])

class TransformerLayersTest(unittest.TestCase):
    max_seq_length = 100
    heads_number=4
    dff=512
    actual_seq_length = 50
    decoder_actual_seq_len = 89
    embedding_size = 512
    lstm_size = 128
    en_vocab_size = 100
    fr_vocab_size = 120

    def test_positionalEncoding_shape(self):
        pos_encoding = PositionalEncodingLayer(embedding_size=self.embedding_size,
                                             max_sentence_len=self.max_seq_length)
        
        input_data = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length, self.embedding_size))
        assert(pos_encoding(input_data).shape == input_data.shape)

    def test_MultiHeadAttentionLayer_shape(self):
        mhaa = MultiHeadAttentionLayer(embedding_size=self.embedding_size,
                                       heads_number=self.heads_number)
        
        q = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length, self.embedding_size))
        k = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length, self.embedding_size))
        v = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length, self.embedding_size))

        x = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length))
        seq_mask = makeSequenceMask(x.shape[1])

        desired_shape = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length, self.embedding_size))

        assert(mhaa(q,k,v,seq_mask).shape == desired_shape.shape)

    def test_EncoderLayer_shape(self):
        encoder_layer = EncoderLayer(embedding_size=self.embedding_size, 
                                    heads_number=self.heads_number,
                                    dff=self.dff)

        input = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length, self.embedding_size))

        x = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length))
        pad_mask = makePaddingMask(x)


        assert(encoder_layer(input, pad_mask, training_enabled=self.training_enabled).shape == input.shape)

    def test_DecoderLayer_shape(self):
        decoder_layer = DecoderLayer(embedding_size=self.embedding_size, 
                                    heads_number=self.heads_number,
                                    dff=self.dff)

        x = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length))
        encoder_output = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length, self.embedding_size))
        decoder_input = tf.random.uniform(shape=(self.batch_size, self.decoder_actual_seq_len, self.embedding_size))

        seq_mask = makeSequenceMask(decoder_input.shape[1])
        pad_mask = makePaddingMask(x)

        assert(decoder_layer.call(decoder_input, encoder_output, pad_mask, seq_mask, training_enabled=self.training_enabled).shape == decoder_input.shape)
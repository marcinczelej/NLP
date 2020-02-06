import sys

import unittest
from parameterized import parameterized_class

import tensorflow as tf

sys.path.insert(0, r"../models")

from TransformerModel import *

@parameterized_class(("batch_size", "training_enabled"), [
    (1, True),
    (1, False),
    (16, True),
    (16, False)
])

class TransformerModelTest(unittest.TestCase):

    max_seq_length = 100
    heads_number=4
    dff=512
    actual_seq_length = 50
    decoder_actual_seq_len = 89
    embedding_size = 512
    lstm_size = 128
    en_vocab_size = 100
    fr_vocab_size = 120
    blocks_amount = 6

    def test_Encoder_shape(self):
        encoder = Encoder(embedding_size=self.embedding_size, 
                         max_sentence_len=self.max_seq_length,
                         vocab_size=self.en_vocab_size,
                         blocks_amount=self.blocks_amount,
                         heads_number=self.heads_number,
                         dff=self.dff)
        

        encoder_input = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length))
        pad_mask = makePaddingMask(encoder_input)


        assert(encoder(encoder_input, pad_mask, self.training_enabled).shape == (*encoder_input.shape, self.embedding_size))
    
    def test_Decoder_shape(self):
        decoder = Decoder(embedding_size=self.embedding_size, 
                         max_sentence_len=self.max_seq_length,
                         vocab_size=self.fr_vocab_size,
                         blocks_amount=self.blocks_amount,
                         heads_number=self.heads_number,
                         dff=self.dff)
        

        encoder_input = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length))
        encoder_output = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length, self.embedding_size))
        decoder_input = tf.random.uniform(shape=(self.batch_size, self.decoder_actual_seq_len))
        pad_mask = makePaddingMask(encoder_input)
        seq_mask = makeSequenceMask(self.decoder_actual_seq_len)


        assert(decoder(encoder_output, decoder_input, pad_mask, seq_mask, self.training_enabled).shape == (*decoder_input.shape, self.embedding_size))
    
    def test_Transformer_shape(self):
        transformer = Transformer(embedding_size=self.embedding_size,
                                  dff=self.dff,
                                  input_max_seq_length=self.max_seq_length,
                                  output_max_seq_length=self.max_seq_length,
                                  input_vocab_size=self.en_vocab_size,
                                  output_vocab_size=self.fr_vocab_size,
                                  encoder_blocks=self.blocks_amount,
                                  decoder_blocks=self.blocks_amount,
                                  heads=self.heads_number)
        

        encoder_input = tf.random.uniform(shape=(self.batch_size, self.actual_seq_length))
        decoder_input = tf.random.uniform(shape=(self.batch_size, self.decoder_actual_seq_len))
        pad_mask = makePaddingMask(encoder_input)
        seq_mask = makeSequenceMask(self.decoder_actual_seq_len)


        assert(transformer(encoder_input, decoder_input, pad_mask, seq_mask, self.training_enabled).shape == (*decoder_input.shape, self.fr_vocab_size))
import sys
import unittest

from parameterized import parameterized

import tensorflow as tf

sys.path.insert(0, r"../models")

from Seq2SeqAttmodel import Encoder, Decoder

class TestSeq2SeqAttentionModel(unittest.TestCase):

    en_vocab_size = 100
    fr_vocab_size = 120
    lstm_size = 128
    en_vocab_size = 100
    fr_vocab_size = 120

    @parameterized.expand([
        (True, 1, "dot"),
        (False, 1, "dot"),
        (True, 1, "general"),
        (False, 1, "general"),
        (True, 1, "concat"),
        (False, 1, "concat"),
        (True, 16, "dot"),
        (False, 16, "dot"),
        (True, 16, "general"),
        (False, 16, "general"),
        (True, 16, "concat"),
        (False, 16, "concat"),
    ])


    def test_encoder_decoder_shapes(self, mode, batch_size, attention_type):
        print("starting Seq2Seq with batch_size = {} with training_mode = {} and attention type = {}" \
            .format(batch_size, mode, attention_type))
        # checks for encoder state

        en_vocab_size = 100
        fr_vocab_size = 120

        encoder = Encoder(lstm_size=self.lstm_size,
                          embedding_size=512,
                          vocab_size=self.en_vocab_size)

        source_input = tf.constant([1, 7, 59, 43, 55, 6, 10, 10])
        source_input = tf.broadcast_to(source_input, [batch_size, source_input.shape[0]])

        initial_state = encoder.init_states(batch_size)
        encoder_output, en_state_h, en_state_c = encoder(source_input, initial_state,
                                                         training_mode=mode)
        
        assert(encoder_output.shape == (*source_input.shape, self.lstm_size))
        assert(en_state_h.shape == (batch_size, self.lstm_size))
        assert(en_state_c.shape == (batch_size, self.lstm_size))


        decoder = Decoder(lstm_size=self.lstm_size,
                          embedding_size=256,
                          vocab_size=self.fr_vocab_size,
                          attention_type=attention_type)

        decoder_input = tf.constant([1])
        decoder_input = tf.broadcast_to(decoder_input, [batch_size, decoder_input.shape[0]])
        decoder_output, de_state_h, de_state_c, alingment_vector = decoder(decoder_input, [en_state_h, en_state_c], encoder_output,
                                                                           training_mode=mode)

        assert(decoder_output.shape == (batch_size, self.fr_vocab_size))
        assert(de_state_h.shape == (batch_size, self.lstm_size))
        assert(de_state_c.shape == (batch_size, self.lstm_size))
        assert(alingment_vector.shape == (batch_size, 1, source_input.shape[1]))
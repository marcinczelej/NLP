import sys

import tensorflow as tf

sys.path.insert(0, r"../models")

from Seq2Seqmodel import Encoder, Decoder

class TestSeq2SeqModel:
    def __init__(self):

        self.en_vocab_size = 100
        self.fr_vocab_size = 120
        self.lstm_size = 128

    def test_encoder_decoder_shapes(self, mode):
        print("starting Seq2Seq with training_mode = ", mode)
        # checks for encoder state

        en_vocab_size = 100
        fr_vocab_size = 120


        batch_size = 16
        encoder = Encoder(lstm_units=self.lstm_size,
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
                          vocab_size=self.fr_vocab_size)

        decoder_input = tf.constant([1,2,3,4,5,6])
        decoder_input = tf.broadcast_to(decoder_input, [batch_size, decoder_input.shape[0]])
        decoder_output, de_state_h, de_state_c = decoder(decoder_input, [en_state_h, en_state_c],
                                                         training_mode=mode)

        assert(decoder_output.shape == (*decoder_input.shape, self.fr_vocab_size))
        assert(de_state_h.shape == (batch_size, self.lstm_size))
        assert(de_state_c.shape == (batch_size, self.lstm_size))

testClass = TestSeq2SeqModel()

testClass.test_encoder_decoder_shapes(mode=True)
testClass.test_encoder_decoder_shapes(mode=False)
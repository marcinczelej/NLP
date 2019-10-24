import tensorflow as tf

# Encoder Decoder network

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_size, units):
    super(Encoder, self).__init__()

    self.units = units
    self.embeding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True, trainable=True)
    self.lstm_layer = tf.keras.layers.LSTM(units, dropout=0.2, return_sequences=True, return_state=True)
  
  def call(self, sequences, lstm_states):
    # sequences shape = [batch_size, seq_max_len]
    # lstm_states = [batch_size, lstm_size] x 2
    # encoder_embedded shape = [batch_size, seq_max_len, embedding_size]
    # output shape = [batch_size, seq_max_len, lstm_size]
    # state_h, state_c shape = [batch_size, lstm_size] x 2

    encoder_embedded = self.embeding_layer(sequences)
    #print("encoder_embedded = ", encoder_embedded.shape)
    output, state_h, state_c = self.lstm_layer(encoder_embedded, initial_state=lstm_states)

    return output, state_h, state_c

  def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.units]),
                tf.zeros([batch_size, self.units]))

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_size, units):
    super(Decoder, self).__init__()

    self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size)
    self.lstm_layer = tf.keras.layers.LSTM(units, dropout=0.2, return_sequences=True,
                                           return_state=True)
    self.dense_layer = tf.keras.layers.Dense(vocab_size)
  
  def call(self, sequences, lstm_states):
    # sequences shape = [batch_size, seq_max_len]
    # embedding shape = [batch_size, seq_max_len, embedding_size]
    # output shape = [batch_szie, seq_max_len, lstm_size]
    # state_h, state_c = [batch_size, lstm_size] x2
    # dense shape = [batch_size, seq_max_len, vocab_size]
    
    decoder_embedded = self.embedding_layer(sequences)
    lstm_output, state_h, state_c = self.lstm_layer(decoder_embedded, lstm_states)
    return self.dense_layer(lstm_output), state_h, state_c

def test_encoder_decoder_shapes():
    # checks for encoder state
    vocab_size = len(en_tokenizer.word_index)+1
    fr_vocab_size = len(fr_tokenizer.word_index)+1
    batch_size = 1
    encoder = Encoder(vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

    source_input = tf.constant([[1, 7, 59, 43, 55, 6, 10, 10]])
    initial_state = encoder.init_states(batch_size)
    encoder_output, en_state_h, en_state_c = encoder(source_input, initial_state)
    
    decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
    decoder_input = tf.constant([[1,2,3,4,5]])
    decoder_output, de_state_h, de_state_c = decoder(decoder_input, [en_state_h, en_state_c])

    assert(decoder_output.shape == (*decoder_input.shape, fr_vocab_size))
    assert(de_state_h.shape == (batch_size, LSTM_SIZE))
    assert(de_state_c.shape == (batch_size, LSTM_SIZE))

    assert(encoder_output.shape == (*source_input.shape, LSTM_SIZE))
    assert(en_state_h.shape == (batch_size, LSTM_SIZE))
    assert(en_state_c.shape == (batch_size, LSTM_SIZE))
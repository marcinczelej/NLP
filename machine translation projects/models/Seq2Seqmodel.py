import tensorflow as tf

# Encoder Decoder network

class Encoder(tf.keras.Model):
  def __init__(self, lstm_units, embedding_size, vocab_size):
    """
      Parameters: 
          lstm_size - number of lstm units
          embedding_size - size of embedding layer
          vocab_size - size of vocabulary for input language
    """

    super(Encoder, self).__init__()

    self.units = lstm_units
    self.embeding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True, trainable=True)
    self.lstm_layer = tf.keras.layers.LSTM(lstm_units, dropout=0.2, return_sequences=True, return_state=True)
  
  def call(self, sequences, lstm_states, training_mode):
    """
      Parameters:
        sequences - tokenized input sequence of shape [batch_size, seq_max_len]
        lstm_states - hidden states of encoder lstm layer of shape 2*[batch_size, lstm_size].
                        Can be get from init_states method of encoder
        training_mode - are we in training or prediction mode. It`s important for dropouts present in lstm_layer
      
      Returns:
        encoder_out - encoder output states for all timesteps of shape [batch_size, seq_max_len, lstm_size]
        state_h, state_c - hidden states of lstm_layer of shape 2*[batch_size, lstm_size]
    """

    # sequences shape = [batch_size, seq_max_len]
    # lstm_states = [batch_size, lstm_size] x 2
    # encoder_embedded shape = [batch_size, seq_max_len, embedding_size]
    # encoder_out shape = [batch_size, seq_max_len, lstm_size]
    # state_h, state_c shape = [batch_size, lstm_size] x 2

    encoder_embedded = self.embeding_layer(sequences, training=training_mode)
    #print("encoder_embedded = ", encoder_embedded.shape)
    encoder_out, state_h, state_c = self.lstm_layer(encoder_embedded, initial_state=lstm_states, training=training_mode)

    return encoder_out, state_h, state_c

  def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.units]),
                tf.zeros([batch_size, self.units]))

class Decoder(tf.keras.Model):
  def __init__(self, lstm_size, embedding_size, vocab_size):
    """
      Parameters: 
          lstm_size - number of lstm units
          embedding_size - size of embedding layer
          vocab_size - size of vocabulary for output language
    """

    super(Decoder, self).__init__()

    self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size)
    self.lstm_layer = tf.keras.layers.LSTM(lstm_size, dropout=0.2, return_sequences=True,
                                           return_state=True)
    self.dense_layer = tf.keras.layers.Dense(vocab_size)
  
  def call(self, sequences, lstm_states, training_mode):
    """
      Parameters:
        sequences - tokenized input sequence of shape [batch_size, seq_max_len]
        lstm_states - hidden states of encoder lstm layer of shape 2*[batch_size, lstm_size].
                        Can be get from init_states method of encoder
        training_mode - are we in training or prediction mode. It`s important for dropouts present in lstm_layer
      
      Returns:
        output_vector - output for given timestep of shape [batch_size, vocab_size]
        state_h, state_c - hidden states of lstm_layer of shape 2*[batch_size, lstm_size]
    """

    # sequences shape = [batch_size, seq_max_len]
    # embedding shape = [batch_size, seq_max_len, embedding_size]
    # output shape = [batch_size, seq_max_len, lstm_size]
    # state_h, state_c = [batch_size, lstm_size] x2
    # dense shape = [batch_size, seq_max_len, vocab_size]
    
    decoder_embedded = self.embedding_layer(sequences, training=training_mode)
    lstm_output, state_h, state_c = self.lstm_layer(decoder_embedded, lstm_states, training=training_mode)
    return self.dense_layer(lstm_output), state_h, state_c
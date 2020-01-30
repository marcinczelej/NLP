import tensorflow as tf

class LuangAttention(tf.keras.Model):
  """
    Class that implements LuangAttention
      - uses current decoder output as input to calculate alligment vector
      - score = h_t_trans*W_a*h_s
      - h_t - decoder hideden_state
      - h_s - encoder_output
      - context_vector = softmax(score)
  """

  def __init__(self, lstm_size, attention_type):
    """
      Parameters: 
          lstm_size - number of lstm units
          attention_type - attention type that should be used. Possible types: dot/general/concat  
    """

    super(LuangAttention, self).__init__()

    self.W_a = tf.keras.layers.Dense(lstm_size, name="LuangAttention_W_a")
    self.W_a_tanh = tf.keras.layers.Dense(lstm_size, activation="tanh", name="LuangAttention_W_a_tanh")
    self.v_a = tf.keras.layers.Dense(1)
    self.type = attention_type
  
  def call(self, decoder_output, encoder_output):
    """
      Method that calculates Attention vectors

      Parameters:
        decoder_output - last output of decoder or starting token for first iteration. Should be of shape:
                          [batch_size, 1, lsts_size]
        encoder_output - hidden states of encoder. Should be of shape: [batch_size, input_seq_max_len, lstm_size]

      Returns:
        context_vector - vector that will be used to calcualte final output of ecoder. Shape [batch_size, 1, lstm_size]
        alignment_vector - vector represents what attention is focusing during given timestep. Shape [batch_size, 1, lstm_size]
    """

    # encoder_output shape [batch_size, seq_max_len, hidden_units_of_encoder]
    # decoder_output shape [batch_size, 1, hidden_units of decoder]
    # score shape [batch_size, 1, seq_max_len]
    if self.type == "dot":
        score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
    elif self.type == "general":
        score = tf.matmul(decoder_output, self.W_a(encoder_output), transpose_b=True)
    elif self.type == "concat":
        decoder_output = tf.broadcast_to(decoder_output, encoder_output.shape)
        concated = self.W_a_tanh(tf.concat((decoder_output, encoder_output), axis=-1))
        score = tf.transpose(self.v_a(concated), [0,2,1])
    else:
        raise Exception("wrong score function selected")
        
    alignment_vector = tf.nn.softmax(score, axis=2)
    context_vector = tf.matmul(alignment_vector, encoder_output)

    return context_vector, alignment_vector

class Encoder(tf.keras.Model):
  def __init__(self, lstm_size, embedding_size, vocab_size):
    """
      Parameters: 
          lstm_size - number of lstm units
          embedding_size - size of embedding layer
          vocab_size - size of vocabulary for input language
    """

    super(Encoder, self).__init__()

    self.units = lstm_size
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, name="Encoder_embedding")
    self.lstm_layer = tf.keras.layers.LSTM(units=lstm_size, dropout=0.2, return_sequences=True, return_state=True, name="Encoder_LSTM")

  def call(self, input_seq, initial_state, training_mode):
    """
      Parameters:
        input_seq - tokenized input sequence of shape [batch_size, seq_max_len]
        initial_state - initial state of encoder lstm layer hidden states of shape [batch_size, lstm_size].
                        Can be get from init_states method of encoder
        training_mode - are we in training or prediction mode. It`s important for dropouts present in lstm_layer
      
      Returns:
        encoder_out - encoder output states for each timestep of shape [batch_size, seq_max_len, lstm_size]
        state_h, state_c - hidden states of lstm_layer of shape 2*[batch_size, lstm_size]
    """

    # input_seq =shape [batch_size, seq_max_len]
    # initial_state shape [batch_size, lstm_hidden_state_size]

    # embedding shape [batch_size, seq_max_len, embedding_size]
    embedded_input = self.embedding(input_seq, training=training_mode)
    #encoder output shape [batch_size, seq_max_len, lstm_size]
    # state_h, state_c shape 2*[batch_size, lstm_size]
    encoder_out, state_h, state_c = self.lstm_layer(inputs=embedded_input, initial_state=initial_state, training=training_mode)

    return encoder_out, state_h, state_c
  
  def init_states(self, batch_size):
    return (tf.zeros([batch_size, self.units]),
            tf.zeros([batch_size, self.units]))

class Decoder(tf.keras.Model):
  def __init__(self, lstm_size, embedding_size, vocab_size, attention_type):
    """
      Parameters: 
          lstm_size - number of lstm units
          embedding_size - size of embedding layer
          vocab_size - size of vocabulary for output language
          attention_type - attention type that should be used. Possible types: dot/general/concat  
    """

    super(Decoder, self).__init__()

    self.units = lstm_size
    self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, name="Decoder_embedding")
    self.lstm_layer = tf.keras.layers.LSTM(lstm_size, dropout=0.2, return_sequences=True, return_state=True, name="Decoder_lstm")
    self.dense_layer = tf.keras.layers.Dense(vocab_size)
    self.attention = LuangAttention(lstm_size, attention_type)

    self.W_c = tf.keras.layers.Dense(lstm_size, activation="tanh", name="Attention_W_c")
    self.W_s = tf.keras.layers.Dense(vocab_size, name="Attenton_W_s")

  def call(self, decoder_input, hidden_states, encoder_output, training_mode):
    """
      Parameters:
        decoder_input - tokenized input element of shape [batch_size, 1]
        hidden_states - hidden states of decoder from last timestep. In case of first call, they are taken form encoder.
                        Shape 2*[batch_size, lstm_size].
        encoder_output - outputs from encoder layer of shape [batch_size, seq_max_len, lstm_size]
        training_mode - are we in training or prediction mode. It`s important for dropouts present in lstm_layer
      
      Returns:
        output_vector - output for given timestep of shape [batch_size, vocab_size]
        state_h, state_c - hidden states of lstm_layer of shape 2*[batch_size, lstm_size]
        alignment - attention vector, that can be used to visualize what we`re focusing each timestep. Shape [batch_size, 1, source_len]
    """
    
    # decoder_input shape [batch_size, 1]
    # hidden_states shape 2*[batch_size, lstm_size]
    # encoder_output shape [batch_size, seq_max_len, lstm_size]
    embedded_input = self.embedding_layer(decoder_input, training=training_mode)
    # embedded_input shape [batch_size, 1, embedding_size]
    # lstm_out shape [batch_size, 1, lstm_size]
    # state_h, state_c shape 2*[batch_size, lstm_size]
    lstm_out, state_h, state_c = self.lstm_layer(embedded_input, hidden_states, training=training_mode)

    # context shape [batch_size, 1 lstm_size]
    # alignment shape [batch_size, 1, source_len]
    context, alignment = self.attention(lstm_out, encoder_output)

    # lstm_out shape [batch_size, lstm_size + lstm_size]
    lstm_out = tf.concat([tf.squeeze(context, axis=1), tf.squeeze(lstm_out, axis=1)], axis=1, name="Decoder_concat")

    # output_vector shape [batch_size, lstm_units]
    output_vector = self.W_c(lstm_out)

    # conversion to vocabulary prob
    # output_vector shape [batch_size, vocab_size]
    output_vector = self.W_s(output_vector)
    return output_vector, state_h, state_c, alignment

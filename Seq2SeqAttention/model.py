import tensorflow as tf

"""
class that implements LuangAttention
  - uses current decoder output as input to calculate alligment vector
  - score = h_t_trans*W_a*h_s
  - h_t - decoder hideden_state
  - h_s - encoder_output
  - context_vector = softmax(score)
"""
class LuangAttention(tf.keras.Model):
  def __init__(self, lstm_size, attention_type):
    super(LuangAttention, self).__init__()

    self.W_a = tf.keras.layers.Dense(lstm_size, name="LuangAttention_W_a")
    self.type = attention_type
  
  def call(self, decoder_output, encoder_output):
    # encoder_output shape [batch_size, seq_max_len, hidden_units_of_encoder]
    # decoder_output shape [batch_size, 1, hidden_units of decoder]
    # score shape [batch_size, 1, seq_max_len]
    if self.type == "dot":
        score = 0
    elif self.type == "general":
        score = tf.matmul(decoder_output, self.W_a(encoder_output), transpose_b=True)
    else:
        raise Exception("not implemented yet")
        
    alignment_vector = tf.nn.softmax(score, axis=2)
    context_vector = tf.matmul(alignment_vector, encoder_output)

    return context_vector, alignment_vector

class Encoder(tf.keras.Model):
  def __init__(self, lstm_units, embedding_size, vocab_size):
    super(Encoder, self).__init__()

    self.units = lstm_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, name="Encoder_embedding")
    self.lstm_layer = tf.keras.layers.LSTM(units=lstm_units, dropout=0.2, return_sequences=True, return_state=True, name="Encoder_LSTM")

  def call(self, input_seq, initial_state):
    # input_seq =shape [batch_size, seq_max_len]
    # initial_state shape [batch_size, lstm_hidden_state_size]

    # embedding shape [batch_size, seq_max_len, embedding_size]
    embedded_input = self.embedding(input_seq)
    #encoder output shape [batch_size, seq_max_len, lstm_size]
    # state_h, state_c shape 2*[batch_size, lstm_size]
    encoder_out, state_h, state_c = self.lstm_layer(inputs=embedded_input, initial_state=initial_state)

    return encoder_out, state_h, state_c
  
  def init_states(self, batch_size):
    return (tf.zeros([batch_size, self.units]),
            tf.zeros([batch_size, self.units]))

class Decoder(tf.keras.Model):
  def __init__(self, lstm_units, embedding_size, vocab_size, attention_type):
    super(Decoder, self).__init__()

    self.units = lstm_units
    self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, name="Decoder_embedding")
    self.lstm_layer = tf.keras.layers.LSTM(lstm_units, dropout=0.2, return_sequences=True, return_state=True, name="Decoder_lstm")
    self.dense_layer = tf.keras.layers.Dense(vocab_size)
    self.attention = LuangAttention(lstm_units, attention_type)

    self.W_c = tf.keras.layers.Dense(lstm_units, activation="tanh", name="Attention_W_c")
    self.W_s = tf.keras.layers.Dense(vocab_size, activation="softmax", name="Attenton_W_s")

  def call(self, decoder_input, hidden_states, encoder_output):
    # decoder_input shape [batch_size, 1]
    # hidden_states shape 2*[batch_size, lstm_size]
    # encoder_output shape [batch_size, seq_max_len, lstm_size]
    embedded_input = self.embedding_layer(decoder_input)
    # embedded_input shape [batch_size, 1, embedding_size]
    # lstm_out shape [batch_size, 1, lstm_size]
    # state_h, state_c shape 2*[batch_szie, lstm_size]
    lstm_out, state_h, state_c = self.lstm_layer(embedded_input, hidden_states)

    # context shape [batch_size, 1 lstm_size]
    # alignment shape [batch_size, 1, source_len]
    context, alignment = self.attention(lstm_out, encoder_output)

    # lstm_out shape [batch_size, lstm_size + lstm_size]
    lstm_out = tf.concat([tf.squeeze(context, axis=1), tf.squeeze(lstm_out, axis=1)], axis=1, name="Decoder_concat")

    # output_vector shape [batch_size, lstm_units]
    output_vector = self.W_c(lstm_out)

    # conversion to vocabulaty prob
    # output_vector shape [batch_size, vocab_size]
    output_vector = self.W_s(output_vector)
    return output_vector, state_h, state_c

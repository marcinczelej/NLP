import numpy as np
import tensorflow as tf

class PositionalEncodingLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_size, max_sentence_len, dtype=tf.float32, **kwargs):
    super(PositionalEncodingLayer, self).__init__(dtype, **kwargs)
    if embedding_size%2 !=0:
      embedding_size+=1
    # embeddiPong size -> depth of model
    # positional encoding should have size : [1, max_sentence_len, embedding_size]
    # 1 is here to make broadcasting possible in call method
    PE = np.zeros((1, max_sentence_len, embedding_size))
    # pos should have shape [1, max_sentence_len] with values <0, max_sentence_len)
    pos = np.arange(start=0, stop=max_sentence_len, step=1)
    pos = pos.reshape(max_sentence_len, 1)
    # i should have shappe [1, embedding_size//2] with values <0, embedding_size//2)
    # we need half of embedding size, because half is needed for each sin/cos 
    # then we put it together into PE and we have [1, max_sentence_len, embedding_size]
    i = np.arange(start=0, stop=embedding_size//2, step=1)
    i = i.reshape(embedding_size//2, 1).T
    PE_sin = np.sin(pos/10000**(2*i/embedding_size))
    PE_cos = np.cos(pos/10000**(2*i/embedding_size))
    # we put sin into even indexes ::2 
    # we put cos into odd indexes, thats why we`re starting from 1 here : 1::2
    PE[0, ::, ::2] = PE_sin
    PE[0, ::, 1::2] = PE_cos
    self.PE = tf.constant(PE, dtype=dtype)
  def getPE(self):
    """
    only for debuging purposes
    """
    return self.PE
  def call(self, inputs):
    """
    inputs shape should be same as self.PE shape
        
      input_shape = tf.shape(inputs)
      return inputs + self.PE[:, :input_shape[-2], :]

    It has to be that way becuase we need to be able to get positional encoding for different lenght 
    for encoder and decoder, when we don`t know max lenght. SO we have to do encoding with bigger buffer
    and take what we need only.

    max_sentence_len in should be bigger or equal as longest input we predict we can get
    """

    input_shape = tf.shape(inputs)
    return inputs + self.PE[:, :input_shape[-2], :input_shape[-1]]

def feedForwardnetwork(dff, d_model):
  """
    Feed Forward network implementation according to paper:
    dff=2048 and d_model =512
    but d_model should be same as embedding_size/d_model in MultiHeadAttention
    ffn(x) = max(0, xW_1 + b+1)W_2 + b_2
    where max(0, ...) -> relu activation
  """
  ffNetwork = tf.keras.Sequential()
  ffNetwork.add(tf.keras.layers.Dense(dff, activation="relu"))
  ffNetwork.add(tf.keras.layers.Dense(d_model))
  return ffNetwork

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_size, heads_number, dtype=tf.float32, **kwargs):
    super(MultiHeadAttentionLayer, self).__init__(dtype=tf.float32, **kwargs)
    """
      return shape : [batch_size, sequence_len, d_model]
      heads_number - tell how many heads will be processed at same time
      d_model - model size ; equal to embedding_size
    """
    self.heads_number = heads_number
    self.d_model = embedding_size
    self.w_q = tf.keras.layers.Dense(self.d_model)
    self.w_k = tf.keras.layers.Dense(self.d_model)
    self.w_v = tf.keras.layers.Dense(self.d_model)

    self.outputLayer = tf.keras.layers.Dense(self.d_model)

  # similar to dot attention but with scaling added
  def ScaledDotProductAttention(self, q, k, v, sequence_mask):
    """
    q shape [batch_size, num_heads, q_seq_len, depth_q]
    k shape [batch_size, num_heads, k_seq_len, depth_k]
    v shape [batch_size, num_heads, v_seq_len, depth_v]

    output contex shape [batch_size, num_heads, q_seq_len, depth_v]
    """
    # matmul(q,k,v)
    # resultion shape [batch_size, num_heads, q_seq_len, k_seq_len]
    qk_matmul = tf.matmul(q, k, transpose_b=True)
    # scaling tf.cast is needed here because tf.sqrt needs float32 type
    # score shape [batch_size, num_heads, q_seq_len, k_seq_len]
    score = qk_matmul/tf.math.sqrt(tf.cast(k.shape[-1], dtype=tf.float32))
    # optional mask
    # mask should be shape [batch_size, num_heads, q_seq_len, k_seq_len]
    # for example [
    #             [0, 1, 1]
    #             [0, 0, 1]
    #             ] shape == (2, 3)
    # we`re adding big negative number, because we only care about present/past words that are przedicted
    if sequence_mask is not None:
      #print(" mask is not none")
      #print("sequence_mask shape {}\nscore shape {}" .format(sequence_mask.shape, score.shape))
      score += sequence_mask*-1e9
    # softmax
    # attention_weights shape [batch_size, num_heads, q_seq_len, k_seq_len]
    attention_weights = tf.nn.softmax(score, axis=-1)
    # matmul(res, V)
    # contex shape [batch_size, num_heads, q_seq_len, depth_v]
    context = tf.matmul(attention_weights, v)
    return context

  def splitHeads(self, data):
    # new shape [batch_size, sequence_len, heads_number, d_model//heads_number]
    data = tf.reshape(data, (data.shape[0], data.shape[1], self.heads_number, data.shape[-1]//self.heads_number))
    # transpose dimentions to [batch_size, heads_number, sequence_len, d_model//heads_number]
    return tf.transpose(data, perm=[0,2,1,3])

  def call(self, q, k, v, sequence_mask):
    """
      q shape [batch_size, sequence_len, d_model]
      k shape [batch_size, sequence_len, d_model]
      v shape [batch_size, sequence_len, d_model]

      after first operations shapes are the same
      next we have to split d_model into heads_number of subbatches
      new shape after reshape only should be : [batch_size, sequence_len, heads_number, d_model//heads_number]
      next shape should be transposed to : [batch_size, heads_number, sequence_len, d_model//heads_number]
      where :
        new_d_model = d_model/heads_number
      
      next make scaled dot-product attention on resulting q,k,v

      next concat returning data to get shape : [batch_size, sequence_len, d_model]
      in order to do this we have to transpose context_vector to get [batch_size, sequence_len, heads_number, d_model//heads_number]

      next put it throug dense layer (d_model) in order to get output
    """
    #print("q shape {}\nk shape {}\n v shape {}" .format(q.shape, k.shape, v.shape))
    q = self.w_q(q)
    k = self.w_k(k)
    v = self.w_v(v)
    #print("AFTER Dense\n  q shape {}\n  k shape {}\n  v shape {}" .format(q.shape, k.shape, v.shape))

    q = self.splitHeads(q)
    k = self.splitHeads(k)
    v = self.splitHeads(v)
    #print("AFTER SPLIT\n  q shape {}\n  k shape {}\n  v shape {}" .format(q.shape, k.shape, v.shape))

    context_vector = self.ScaledDotProductAttention(q, k, v, sequence_mask)
    #print("context_vector shape :", context_vector.shape)

    context_vector = tf.transpose(context_vector, perm=[0,2,1,3])
    #print("context_vector  transposed shape :", context_vector.shape)
    context_vector = tf.reshape(context_vector, (context_vector.shape[0], context_vector.shape[1], self.d_model))
    #print("context_vector  reshapeed shape :", context_vector.shape)

    return self.outputLayer(context_vector)
    
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_size, heads_number, dff, dtype=tf.float32, **kwargs):
    """
      Parameters:
        embedding_size - embeddint size ( d_model )
        heads_number - number of attention heads
        dff - dimension of first dense layer for feed forward neural network
      
    """

    super(EncoderLayer, self).__init__(dtype, **kwargs)

    self.d_model = embedding_size
    self.multiHeadAttention = MultiHeadAttentionLayer(embedding_size, heads_number)

    self.normalizationFirst = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.normalizationSecond = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout = tf.keras.layers.Dropout(0.2)

    self.ffNetwork = feedForwardnetwork(dff, self.d_model)

  def call(self, encoder_input, mask, training_enabled):
    """
      Parameters:
        encoder_input - encoder layer input data of shape [batch_size, max_sentence_len, embedding_size]
        mask - mask that will be passed to MultiheadAttention layer, that is used for scaledDotAttention
        training_enabled - are we in training or prediction mode. It`s important for dropouts present in lstm_layer
      
      Returns:
        feedforward network output - of shape [batch_size, seq_max_len, embedding_size]
    """

    # shortcut_data shape [batch_size, max_sentence_len, embedding_size]
    shortcut_data = encoder_input

    # mhatt_output shape [batch_size, max_sentence_len, embedding_size]
    mhatt_output = self.multiHeadAttention(encoder_input, encoder_input, encoder_input, mask)
    mhatt_output = self.dropout(mhatt_output, training=training_enabled)
    mhatt_output += shortcut_data
    mhatt_output = self.normalizationFirst(mhatt_output)

    shortcut_data = mhatt_output

    ffNet_output = self.ffNetwork(mhatt_output)
    ffNet_output = self.dropout(ffNet_output, training=training_enabled)
    ffNet_output += shortcut_data
    ffNet_output = self.normalizationSecond(ffNet_output)

    return ffNet_output

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_size, heads_number, dff, dtype=tf.float32, **kwargs):
    """
      Parameters:
        embedding_size - embeddint size ( d_model )
        heads_number - number of attention heads
        dff - dimension of first dense layer for feed forward neural network
    """

    super(DecoderLayer, self).__init__(dtype, **kwargs)

    self.d_model = embedding_size
    self.multiHeadAttentionFirst = MultiHeadAttentionLayer(embedding_size, heads_number)
    self.multiHeadAttentionSecond = MultiHeadAttentionLayer(embedding_size, heads_number)

    self.normalizationFirst = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.normalizationSecond = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.normalizationThird = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout = tf.keras.layers.Dropout(0.2)


    self.ffNetwork = feedForwardnetwork(dff, self.d_model)

  def call(self, decoder_input, encoder_output, pad_mask, elements_mask, training_enabled):
    """
      Parameters:
        decoder_input - input data of decoder of shape [batch_szie, max_sentence_len, embedding_size]
        encoder_output - output of shape [batch_size, seq_max_len, embedding_size]
        pad_mask - mask that is used for padding tokens masking
        elements_mask - mask that`s used for masking elements past current one
        training_enabled - are we in training or prediction mode. It`s important for dropouts present in lstm_layer
      
      Returns:
        feedforward network output - of shape [batch_size, seq_max_len, embedding_size]
    """

    # shortcut_data shape [batch_szie, max_sentence_len, embedding_size]
    shortcut_data = decoder_input
      
    # mhatt_output shape [batch_size, max_sentence_len, embedding_size]
    mhatt_output = self.multiHeadAttentionFirst(decoder_input, decoder_input, decoder_input, elements_mask)
    mhatt_output = self.dropout(mhatt_output, training=training_enabled)
    # add & Norm
    mhatt_output += shortcut_data
    mhatt_output = self.normalizationFirst(mhatt_output)

    shortcut_data = mhatt_output
    #print("decoder_input ", decoder_input.shape)
    #print("encoder_output ", encoder_output.shape)
    #print("mhatt_output ", mhatt_output.shape)
    mhatt_output2 = self.multiHeadAttentionSecond(mhatt_output, encoder_output, encoder_output, pad_mask)
    mhatt_output2 = self.dropout(mhatt_output2, training=training_enabled)
    mhatt_output2 += shortcut_data
    mhatt_output2 = self.normalizationSecond(mhatt_output2)

    shortcut_data = mhatt_output2
    ffn_output = self.ffNetwork(mhatt_output2)
    ffn_output = self.dropout(ffn_output, training=training_enabled)
    ffn_output += shortcut_data
    ffNet_output = self.normalizationThird(ffn_output)

    return ffNet_output
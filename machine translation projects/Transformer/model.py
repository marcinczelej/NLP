import numpy as np
import tensorflow as tf

def loadWeights(transformer_model, checkpointPath):
    ckpt = tf.train.Checkpoint(transformer=transformer_model)

    manager = tf.train.CheckpointManager(ckpt, checkpointPath, max_to_keep=5)

    if manager.latest_checkpoint:
      ckpt.restore(manager.latest_checkpoint).expect_partial()
      print ('Latest checkpoint restored!!')
    else:
      print("Failed to restore checkpoint")
    
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

def feedForwardnetwork(dff, d_model):
  """
  according to paper dff=2048 and d_model =512
  but d_model should be same as embedding_size/d_model in MultiHeadAttention
  ffn(x) = max(0, xW_1 + b+1)W_2 + b_2
  where max(0, ...) -> relu activation
  """
  ffNetwork = tf.keras.Sequential()
  ffNetwork.add(tf.keras.layers.Dense(dff, activation="relu"))
  ffNetwork.add(tf.keras.layers.Dense(d_model))
  return ffNetwork

def makeSequenceMask(seq_len):
  """
  mask should be size [1, 1, seq_len, seq_len]
  first two sizes are batch_szie, num_heads to make this matrix broadcastable
  it should be in form 
  [
    [0, 1, 1, 1]
    [0, 0, 1, 1]
    [0, 0, 0, 1]
    [0, 0, 0, 0]
  ]
  """
  mask_array = np.ones((seq_len, seq_len))
  mask_array = np.triu(mask_array, 1)
  return tf.constant(mask_array, dtype=tf.float32)

def makePaddingMask(sequence):
  mask = tf.math.equal(sequence, 0)
  mask =  tf.cast(mask, tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_size, heads_number, dff, dtype=tf.float32, **kwargs):
    super(EncoderLayer, self).__init__(dtype, **kwargs)

    self.d_model = embedding_size
    self.multiHeadAttention = MultiHeadAttentionLayer(embedding_size, heads_number)

    self.normalizationFirst = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.normalizationSecond = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropoutFirst = tf.keras.layers.Dropout(0.2)
    self.dropoutSecond = tf.keras.layers.Dropout(0.2)

    self.ffNetwork = feedForwardnetwork(dff, self.d_model)

  def call(self, encoder_input, mask, training_enabled):
    # shortcut_data shape [batch_size, max_sentence_len, embedding_size]
    shortcut_data = encoder_input

    # mhatt_output shape [batch_size, max_sentence_len, embedding_size]
    mhatt_output = self.multiHeadAttention(encoder_input, encoder_input, encoder_input, mask)
    mhatt_output = self.dropoutFirst(mhatt_output, training=training_enabled)
    mhatt_output += shortcut_data
    mhatt_output = self.normalizationFirst(mhatt_output)

    shortcut_data = mhatt_output

    ffNet_output = self.ffNetwork(mhatt_output)
    ffNet_output = self.dropoutSecond(ffNet_output, training=training_enabled)
    ffNet_output += shortcut_data
    ffNet_output = self.normalizationSecond(ffNet_output)

    return ffNet_output

class Encoder(tf.keras.Model):
  """
  encoder_input shape [batch_size, max_sentence_len]
  
  Encoder flow :

  - Embedding 
  - Positional Encoding
  - Input = Embedding + Positional Encoding
  --------------------REPEAT N Times--------------------
  - Multi-head Attention layer
  - Input + Multi-Head Attention layer added together 
  - previous Normalized (1)
  - Feed Forward Network (2)
  - (1) added to (2) and Normmalized
  ------------------------------------------------------
  - Encoder output 
  """
  def __init__(self, embedding_size, max_sentence_len, vocab_size, blocks_amount, heads_number, dff):
    super(Encoder, self).__init__()

    assert (embedding_size//heads_number)%2==0
    self.blocks_amount = blocks_amount
    self.d_model = embedding_size
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
    self.positionalEncoding = PositionalEncodingLayer(embedding_size, max_sentence_len)

    self.encoderBlocks = [EncoderLayer(embedding_size, heads_number, dff) for _ in range(blocks_amount)]
  
  def call(self, encoder_input, mask, training_enabled=False):
    # sequence shape [batch_size, max_sentence_len]
    embedded_seq = self.embedding(encoder_input)
    # according to paper https://arxiv.org/pdf/1706.03762.pdf
    # embedding is multiplied by sqrt(d_model). Point 3.4
    embedded_seq*=tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # embedded_seq shape [batch_szie, max_sentence_len, embedding_size]
    data = self.positionalEncoding(embedded_seq)
    #------------------------- loop though all blocks -------------------------
    for i in range(self.blocks_amount):
      #print("               BLOCK ", i+1)
      data = self.encoderBlocks[i](data, mask, training_enabled) 

    return data

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, embedding_size, heads_number, dff, dtype=tf.float32, **kwargs):
    super(DecoderLayer, self).__init__(dtype, **kwargs)

    self.d_model = embedding_size
    self.multiHeadAttentionFirst = MultiHeadAttentionLayer(embedding_size, heads_number)
    self.multiHeadAttentionSecond = MultiHeadAttentionLayer(embedding_size, heads_number)

    self.normalizationFirst = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.normalizationSecond = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.normalizationThird = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropoutFirst = tf.keras.layers.Dropout(0.2)
    self.dropoutSecond = tf.keras.layers.Dropout(0.2)
    self.dropoutThird = tf.keras.layers.Dropout(0.2)

    self.ffNetwork = feedForwardnetwork(dff, self.d_model)

  def call(self, decoder_input, encoder_output, pad_mask, elements_mask, training_enabled):
    # shortcut_data shape [batch_szie, max_sentence_len, embedding_size]
    shortcut_data = decoder_input
      
    # mhatt_output shape [batch_size, max_sentence_len, embedding_size]
    mhatt_output = self.multiHeadAttentionFirst(decoder_input, decoder_input, decoder_input, elements_mask)
    mhatt_output = self.dropoutFirst(mhatt_output, training=training_enabled)
    # add & Norm
    mhatt_output += shortcut_data
    mhatt_output = self.normalizationFirst(mhatt_output)

    shortcut_data = mhatt_output
    #print("decoder_input ", decoder_input.shape)
    #print("encoder_output ", encoder_output.shape)
    #print("mhatt_output ", mhatt_output.shape)
    mhatt_output2 = self.multiHeadAttentionSecond(mhatt_output, encoder_output, encoder_output, pad_mask)
    mhatt_output2 = self.dropoutSecond(mhatt_output2, training=training_enabled)
    mhatt_output2 += shortcut_data
    mhatt_output2 = self.normalizationSecond(mhatt_output2)

    shortcut_data = mhatt_output2
    ffn_output = self.ffNetwork(mhatt_output2)
    ffn_output = self.dropoutThird(ffn_output, training=training_enabled)
    ffn_output += shortcut_data
    ffNet_output = self.normalizationThird(ffn_output)

    return ffNet_output

class Decoder(tf.keras.models.Model):
  """
  Decoder flow :

  - Embedding 
  - Positional Encoding
  - Input = Embedding + Positional Encoding
  --------------------REPEAT N Times--------------------
  - Masked Multi-head Attention layer with elements_mask
  - Input + Masked Multi-Head Attention layer added together 
  - previous Normalized (1) 
  - Multi-head Attention layer v, k from Encoder output | q from previous point with padding mask
  - (1) + Multi-head Attention layer added together
  - previous normalized
  - Feed Forward Network (2)
  - (1) added to (2) and Normalized
  ------------------------------------------------------
  - Decoder output

  decoder masks are :
    - encoder_padding_mask - padding mask made on encoder input data
    - decoder sequences mask - sequence mask made on decoder input data
  """
  def __init__(self, embedding_size, max_sentence_len, vocab_size, blocks_amount, heads_number, dff):
    super(Decoder, self).__init__()

    assert (embedding_size//heads_number)%2==0
    self.blocks_amount = blocks_amount
    self.d_model = embedding_size
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
    self.positionalEncoding = PositionalEncodingLayer(embedding_size, max_sentence_len)
    self.dropout = tf.keras.layers.Dropout(0.1)

    self.decoderBlocks = [DecoderLayer(embedding_size, heads_number, dff) for _ in range(blocks_amount)]

  def call(self, encoder_output, decoder_input, pad_mask, elements_mask, training_enabled=False):

    # sequence shape [batch_size, max_sentence_len]
    embedded_seq = self.embedding(decoder_input)
    # according to paper https://arxiv.org/pdf/1706.03762.pdf
    # embedding is multiplied by sqrt(d_model). Point 3.4
    embedded_seq*=tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # embedded_seq shape [batch_szie, max_sentence_len, embedding_size]
    data = self.positionalEncoding(embedded_seq)
    data = self.dropout(data)
    #------------------------- loop though all blocks -------------------------
    for i in range(self.blocks_amount):
      #print("               BLOCK ", i+1)
      data = self.decoderBlocks[i](data, encoder_output, pad_mask, elements_mask, training_enabled)

    return data

class Transformer(tf.keras.models.Model):
  """
  Transformer flow:

  - Encoder
  - Decoder
  - Dense

   transformer_out shape = [batch_size, output_seq_len, output_vocab_size]
   default trainng_enabled == False
  """
  def __init__(self,
               embedding_size,
               dff,
               input_max_seq_length,
               output_max_seq_length,
               input_vocab_size,
               output_vocab_size,
               encoder_blocks,
               decoder_blocks,
               heads):
    super(Transformer, self).__init__()

    self.encoder = Encoder(embedding_size, input_max_seq_length, input_vocab_size, encoder_blocks, heads, dff)
    self.decoder = Decoder(embedding_size, output_max_seq_length, output_vocab_size, decoder_blocks, heads, dff)

    self.dense = tf.keras.layers.Dense(output_vocab_size)

  def call(self, input_seq, output_seq, pad_mask, words_mask, training_enabled=False):
    
    encoder_out = self.encoder(input_seq,
                               mask=pad_mask,
                               training_enabled=training_enabled)
    decoder_out = self.decoder(encoder_out,
                               output_seq,
                               pad_mask=pad_mask,
                               elements_mask=words_mask,
                               training_enabled=training_enabled)

    transformer_out = self.dense(decoder_out)
    return transformer_out
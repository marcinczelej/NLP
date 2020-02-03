import numpy as np
import tensorflow as tf

from TransformerLayers import *

def loadWeights(transformer_model, checkpointPath):
    ckpt = tf.train.Checkpoint(transformer=transformer_model)

    manager = tf.train.CheckpointManager(ckpt, checkpointPath, max_to_keep=5)

    if manager.latest_checkpoint:
      ckpt.restore(manager.latest_checkpoint).expect_partial()
      print ('Latest checkpoint restored!!')
    else:
      print("Failed to restore checkpoint")

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
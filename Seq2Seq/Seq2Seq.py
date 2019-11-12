import os
import sys
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, r"../utilities")

from utils import *
from model import Decoder, Encoder

LSTM_SIZE = 512
EMBEDDING_SIZE = 250
BATCH_SIZE= 64
EPOCHS = 600

def main():
    data_dir = "../data"
    data = read_data(os.path.join(data_dir, "fra-eng"), "fra.txt")
    en_lines, fr_lines = zip(*data)

    en_lines = [normalize(line) for line in en_lines]
    fr_lines = [normalize(line) for line in fr_lines]

    fr_train, fr_test, en_train, en_test = train_test_split(fr_lines, en_lines, shuffle=True, test_size=0.1)

    fr_train_in = ['<start> ' + line for line in fr_train]
    fr_train_out = [line + ' <end>' for line in fr_train]

    fr_tokenizer = Tokenizer(filters='')
    en_tokenizer = Tokenizer(filters='')

    input_data = [fr_train_in, fr_train_out, fr_test]
    fr_train_in, fr_train_out, fr_test = tokenizeInput(input_data, fr_tokenizer)

    input_data = [en_train, en_test]
    en_train, en_test = tokenizeInput(input_data, en_tokenizer)
    
    strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = BATCH_SIZE*strategy.num_replicas_in_sync

print("creating dataset...")
train_dataset = tf.data.Dataset.from_tensor_slices((en_train, fr_train_in, fr_train_out))
train_dataset = train_dataset.shuffle(len(en_train)).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
train_dataset = strategy.experimental_distribute_dataset(train_dataset)

print("dataset created")
print("batches each epoch : ", len(en_train)/BATCH_SIZE)
min_loss = 1000000

vocab_size = len(en_tokenizer.word_index)+1
fr_vocab_size = len(fr_tokenizer.word_index)+1

with strategy.scope():
    optim = tf.keras.optimizers.Adam(clipnorm=5.0)
    encoder = Encoder(vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
    decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
    
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE) # output is softmax result
    def compute_loss(predictions, labels):
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        mask = tf.cast(mask, tf.int64)
        per_example_loss = loss_obj(labels, predictions, sample_weight=mask)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    
    # predicting random sentence output
    def predict_output():
      index = np.random.choice(len(en_test))
      en_sentence = en_test[index]
      should_be_sentence = fr_test[index]

      sentence = en_tokenizer.texts_to_sequences([en_sentence])
      initial_states = encoder.init_states(1)
      _, state_h, state_c = encoder(tf.constant(sentence), initial_states, training=False)

      symbol = tf.constant([[fr_tokenizer.word_index['<start>']]])
      sentence = []

      while True:
        symbol, state_h, state_c = decoder(symbol, (state_h, state_c), training=False)
        # argmax to get max index 
        symbol = tf.argmax(symbol, axis=-1)
        word = fr_tokenizer.index_word[symbol.numpy()[0][0]]

        if len(sentence) >=23 or word == '<end>':
          break

        sentence.append(word + " ")

      predicted_sentence = ''.join(sentence)
      print("--------------PREDICTION--------------")
      print("Predicted sentence:  {} " .format(predicted_sentence))
      print("Should be sentence:  {} " .format(should_be_sentence))
      print("------------END PREDICTION------------")

    # one training step
    def train_step(encoder_input, decoder_in, decoder_out, initial_states):
        with tf.GradientTape() as tape:
            encoder_states = encoder(encoder_input, initial_state, training=True)
            predictions, _, _ = decoder(decoder_in, encoder_states[1:], training=True)
            loss = compute_loss(predictions, decoder_out)
  
        trainable = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(loss, trainable)
        optim.apply_gradients(zip(grads, trainable))
        return loss
    
    @tf.function
    def distributed_train_step(encoder_input, decoder_in, decoder_out, initial_states):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                      args=(encoder_input,
                                                            decoder_in,
                                                            decoder_out,
                                                            initial_states,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)

    for epoch in range(EPOCHS):
        initial_state = encoder.init_states(BATCH_SIZE)
        total_loss = 0.0
        num_batches = 0

        for batch_nr, (en_data, fr_data_in, fr_data_out) in enumerate(train_dataset):
            single_loss = distributed_train_step(en_data, fr_data_in, fr_data_out, initial_state)
            total_loss += single_loss
            num_batches += 1

        loss = total_loss/num_batches
        print(" EPOCH : {} loss {} " .format(epoch, loss))
        if loss < min_loss:
            print("saving weights in epoch ", epoch)
            encoder.save_weights('saved_models/best_encoder_weights.h5')
            decoder.save_weights('saved_models/best_decoder_weights.h5')
            min_loss = loss

        try:
            predict_output()
        except Exception:
            continue

if __name__ == "__main__":
    main()
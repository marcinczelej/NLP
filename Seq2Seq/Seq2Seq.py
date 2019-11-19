import os
import sys
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, r"../utilities")

from utils import *
from model import Encoder, Decoder
from Seq2SeqTrainer import Seq2SeqTrainer

LSTM_SIZE = 512
EMBEDDING_SIZE = 250
BATCH_SIZE= 64
EPOCHS = 600

def main():
    data_dir = "../data"
    # reading data

    en_lines, fr_lines = read_data_files(data_dir, ("small_vocab_en", "small_vocab_fr"))
    """
    data = read_data(os.path.join(data_dir, "fra-eng"), "fra.txt")
    en_lines, fr_lines = list(zip(*data))
    
    en_lines = en_lines[:30000]
    fr_lines = fr_lines[:30000]
    """
    en_lines = [normalize(line) for line in en_lines]
    fr_lines = [normalize(line) for line in fr_lines]

    en_train, en_test, fr_train, fr_test = train_test_split(en_lines, fr_lines, shuffle=True, test_size=0.1)

    fr_train_in = ['<start> ' + line for line in fr_train]
    fr_train_out = [line + ' <end>' for line in fr_train]

    fr_test_in = ['<start> ' + line for line in fr_test]
    fr_test_out = [line + ' <end>' for line in fr_test]

    fr_tokenizer = Tokenizer(filters='')
    en_tokenizer = Tokenizer(filters='')

    input_data = [fr_train_in, fr_train_out, fr_test_in, fr_test_out, fr_test, fr_train]
    fr_train_in, fr_train_out, fr_test_in, fr_test_out, fr_test, fr_train = tokenizeInput(input_data,
                                                                                          fr_tokenizer)

    input_data = [en_train, en_test]
    en_train, en_test = tokenizeInput(input_data, en_tokenizer)

    en_vocab_size = len(en_tokenizer.word_index)+1
    fr_vocab_size = len(fr_tokenizer.word_index)+1
    print("en_vocab {}\nfr_vocab {}" .format(en_vocab_size, fr_vocab_size))
    
    trainer = Seq2SeqTrainer(BATCH_SIZE, LSTM_SIZE, EMBEDDING_SIZE, predict_every=1)
    trainer.train([en_train, fr_train_in, fr_train_out], [en_test, fr_test_in, fr_test_out], [en_tokenizer, fr_tokenizer], 20)
    
    """
    strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = BATCH_SIZE*strategy.num_replicas_in_sync

    print("creating dataset...")
    train_dataset = tf.data.Dataset.from_tensor_slices((en_train, fr_train_in, fr_train_out))
    train_dataset = train_dataset.shuffle(len(en_train)).batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    train_dataset_distr = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((en_test, fr_test_in, fr_test_out))
    test_dataset = test_dataset.shuffle(len(en_test), reshuffle_each_iteration=True)\
                                   .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    test_dataset_distr = strategy.experimental_distribute_dataset(test_dataset)

    print("dataset created")
    print("batches each epoch : ", len(en_train)/BATCH_SIZE)
    min_loss = 1000000
    
    test_losses = []
    train_losses = []
    train_accuracyVec = []
    test_accuracyVec =[]
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    one_step_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    with strategy.scope():
        optim = tf.keras.optimizers.Adam(clipnorm=5.0)
        encoder = Encoder(en_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)
        decoder = Decoder(fr_vocab_size, EMBEDDING_SIZE, LSTM_SIZE)

        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                                 reduction="none") 
        def compute_loss(predictions, labels):
            mask = tf.math.logical_not(tf.math.equal(labels, 0))
            mask = tf.cast(mask, tf.int64)
            per_example_loss = loss_obj(labels, predictions, sample_weight=mask)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        def predict(input_data, real_data_out):
          en_sentence = en_tokenizer.sequences_to_texts([input_data])
          input_data = tf.expand_dims(input_data, 0)
          initial_states = encoder.init_states(1)
          _, state_h, state_c = encoder(tf.constant(input_data), initial_states, training=False)
          
          symbol = tf.constant([[fr_tokenizer.word_index['<start>']]])
          sentence = []

          while True:
            symbol, state_h, state_c = decoder(symbol, (state_h, state_c), training=False)
            # argmax to get max index 
            symbol = tf.argmax(symbol, axis=-1)
            word = fr_tokenizer.index_word[symbol.numpy()[0][0]]

            if word == '<end>' or len(sentence) >= len(real_data_out):
              break
            
            sentence.append(word)
          print("--------------PREDICTION--------------")
          print("  English   :  {}" .format(en_sentence))
          print("  Predicted :  {}" .format(' '.join(sentence)))
          print("  Correct   :  {}" .format(fr_tokenizer.sequences_to_texts([real_data_out])))
          print("------------END PREDICTION------------")

        # one training step
        def train_step(encoder_input, decoder_in, decoder_out, initial_states):
            with tf.GradientTape() as tape:
                encoder_states = encoder(encoder_input, initial_state, training=True)
                predicted_data, _, _ = decoder(decoder_in, encoder_states[1:], training=True)
                loss = compute_loss(predicted_data, decoder_out)

            trainable = encoder.trainable_variables + decoder.trainable_variables
            grads = tape.gradient(loss, trainable)
            optim.apply_gradients(zip(grads, trainable))
            train_accuracy.update_state(decoder_out, predicted_data)
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
        
        def test_step(encoder_input, decoder_in, decoder_out):
            initial_state = encoder.init_states(BATCH_SIZE)
            encoder_states = encoder(encoder_input, initial_state, training=False)
            predicted_data, _, _ = decoder(decoder_in, encoder_states[1:], training=False)
            loss = compute_loss(predicted_data, decoder_out)

            train_accuracy.update_state(decoder_out, predicted_data)
            return loss
        
        @tf.function
        def distributed_test_step(encoder_input, decoder_in, decoder_out):
            per_replica_losses = strategy.experimental_run_v2(test_step,
                                                          args=(encoder_input,
                                                                decoder_in,
                                                                decoder_out,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)
        for epoch in range(EPOCHS):
            test_accuracy.reset_states()
            train_accuracy.reset_states()
            initial_state = encoder.init_states(BATCH_SIZE)
            total_loss = 0.0
            num_batches = 0
            print("train_step")
            for _, (en_data, fr_data_in, fr_data_out) in enumerate(train_dataset_distr):
                loss = distributed_train_step(en_data, fr_data_in, fr_data_out, initial_state)
                total_loss += loss
                num_batches += 1
            train_losses.append(total_loss/num_batches)
            total_loss = 0.0
            num_batches = 0
            print("test_Step")
            for _, (en_data, fr_data_in, fr_data_out) in enumerate(test_dataset_distr):
                loss = distributed_test_step(en_data, fr_data_in, fr_data_out)
                total_loss += loss
                num_batches += 1
            test_losses.append(total_loss/num_batches)
                
            print ('Epoch {} training Loss {:.4f} Accuracy {:.4f}  test Loss {:.4f} Accuracy {:.4f}' .format( \
                                                  epoch + 1, 
                                                  train_losses[-1], 
                                                  train_accuracy.result(),
                                                  test_losses[-1],
                                                  test_accuracy.result()))
        try:
            idx = np.random.randint(low=0, high=len(en_test), size=1)[0]
        except:
            print("exception")
        predict(en_test[idx], fr_test[idx])
        """
if __name__ == "__main__":
    main()
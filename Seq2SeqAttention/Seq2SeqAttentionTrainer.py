import numpy as np
import tensorflow as tf

from model import Encoder, Decoder

class Seq2SeqAttentionTrainer:
    def __init__(self, batch_size, lstm_size, embedding_size, predict_every):
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.predict_every = predict_every
        self.strategy = tf.distribute.MirroredStrategy()
        self.encoder = None
        self.decoder = None
        self.optimizer = None
        self.fr_tokenizer = None
        self.en_tokenizer = None

    def train(self, train_dataset_data, test_dataset_data, tokenizers, epochs, attention_type):
        """
            train_dataset_data should be made from (en_train, fr_train_in, fr_train_out)
            test_dataset_data should be made from (en_test, fr_test_in, fr_test_out)
        """
        
        self.en_tokenizer, self.fr_tokenizer = tokenizers
        en_vocab_size = len(self.en_tokenizer.word_index)+1
        fr_vocab_size = len(self.fr_tokenizer.word_index)+1
        print("en_vocab {}\nfr_vocab {}" .format(en_vocab_size, fr_vocab_size))
        
        print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
        GLOBAL_BATCH_SIZE = self.batch_size*self.strategy.num_replicas_in_sync

        print("creating dataset...")
        en_train, fr_train_in, fr_train_out = train_dataset_data
        en_test, fr_test_in, fr_test_out = test_dataset_data
        train_dataset = tf.data.Dataset.from_tensor_slices((en_train, fr_train_in, fr_train_out))
        train_dataset = train_dataset.shuffle(len(en_train), reshuffle_each_iteration=True)\
                                        .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

        test_dataset = tf.data.Dataset.from_tensor_slices((en_test, fr_test_out))
        test_dataset = test_dataset.shuffle(len(en_test), reshuffle_each_iteration=True)\
                                       .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        test_dist_dataset = self.strategy.experimental_distribute_dataset(test_dataset)
        print("dataset created")
        
        test_losses = []
        train_losses = []
        train_accuracyVec = []
        test_accuracyVec =[]
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        one_step_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        min_test_loss = 10000.0

        with self.strategy.scope():
            self.encoder = Encoder(self.lstm_size, self.embedding_size, en_vocab_size)
            self.decoder = Decoder(self.lstm_size, self.embedding_size, fr_vocab_size, attention_type)
            self.optimizer = tf.keras.optimizers.Adam(clipnorm=0.5)

            def predict_output(en_sentence, fr_sentence):
                should_be_sentence = fr_sentence
                sentence = self.en_tokenizer.texts_to_sequences([en_sentence])

                initial_states = self.encoder.init_states(1)
                encoder_out, state_h, state_c = self.encoder(tf.constant(sentence), initial_states, training=False)

                decoder_in = tf.constant([[self.fr_tokenizer.word_index['<start>']]])
                sentence = []
                while True:
                    decoder_out, state_h, state_c = self.decoder( \
                                 decoder_in, (state_h, state_c), encoder_out, training=False)
                    # argmax to get max index 
                    decoder_in = tf.expand_dims(tf.argmax(decoder_out, -1), 0)
                    word = self.fr_tokenizer.index_word[decoder_in.numpy()[0][0]]

                    if  word == '<end>':
                        break
                    sentence.append(word)

                predicted_sentence = ' '.join(sentence)
                print("----------------------------PREDICTION----------------------------")
                print("       En sentence {} " .format(en_sentence))
                print("       Predicted:  {} " .format(predicted_sentence))
                print("       Should be:  {} " .format(should_be_sentence))
                print("--------------------------END PREDICTION--------------------------")

            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, reduction="none")
            def compute_loss(predictions, labels):
                mask = tf.math.logical_not(tf.math.equal(labels, 0))
                mask = tf.cast(mask, tf.int64)
                per_example_loss = loss_obj(labels, predictions, sample_weight=mask)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
            
            # one training step
            def train_step(en_data, fr_data_in, fr_data_out, initial_states):
                one_step_test_accuracy.reset_states()
                with tf.GradientTape() as tape:
                    encoder_output, state_h, state_c = self.encoder(en_data, initial_states, training=True)
                    # shape[1] because we want each word for all batches
                    for i in range(fr_data_out.shape[1]):
                        decoder_input = tf.expand_dims(fr_data_in[:,i], 1)
                        decoder_output, state_h, state_c = self.decoder(decoder_input,
                                                                        (state_h, state_c),
                                                                        encoder_output,
                                                                        training=True)
                        print("decoder_output ", decoder_output)
                        print("fr_data_out[:,i] ", fr_data_out[:,i])
                        loss +=compute_loss(decoder_output, fr_data_out[:,i])
                        #one_step_test_accuracy.update_states(decoder_output, fr_data_out[:,i])

                trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients(zip(grads, trainable_vars))

                train_accuracy.update_states(one_step_test_accuracy.result())
                return loss / fr_data_out.shape[1]

            @tf.function
            def distributed_train_step(en_data, fr_data_in, fr_data_out, initial_states):
                per_replica_losses = self.strategy.experimental_run_v2(train_step,
                                                                  args=(en_data,
                                                                        fr_data_in,
                                                                        fr_data_out,
                                                                        initial_states,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            def test_step(en_data, fr_data_out):
                one_step_test_accuracy.reset_states()
                initial_states = self.encoder.init_states(self.batch_size)
                encoder_output, state_h, state_c = self.encoder(en_data, initial_states, training=False)

                decoder_input = tf.constant(self.fr_tokenizer.word_index['<start>'], shape=(self.batch_size, 1))

                for i in range(fr_data_out.shape[1]): 
                    decoder_output, state_h, state_c = self.decoder(decoder_input,
                                                                    (state_h, state_c),
                                                                    encoder_output,
                                                                    training=False)
                    decoder_input =tf.expand_dims(tf.argmax(decoder_output, 1),1)
                    loss +=compute_loss(decoder_output, fr_data_out[:,i])
                    one_step_test_accuracy.update_states(decoder_output, fr_data_out[:,i])
                
                train_accuracy.update_states(one_step_test_accuracy.result())
                return loss/fr_data_tokenized.shape[1]

            @tf.function
            def distributed_test_step(en_data, fr_data_out):
                per_replica_losses = self.strategy.experimental_run_v2(test_step,
                                                                 args=(en_data,
                                                                       fr_data_out,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            for epoch in range(epochs):
                test_accuracy.reset_states()
                train_accuracy.reset_states()
                initial_states = self.encoder.init_states(self.batch_size)
                
                total_loss = 0.0
                num_batches = 0
                for _, (en_data, fr_data_in, fr_data_out) in enumerate(train_dist_dataset):
                    loss = distributed_train_step(en_data, fr_data_in, fr_data_out, initial_states)
                    total_loss += loss
                    num_batches += 1
                train_losses.append(total_loss/num_batches)
                total_loss = 0.0
                num_batches = 0
                for _, (en_data, fr_data_out) in enumerate(test_dist_dataset):
                    loss = distributed_test_step(en_data, fr_data_out)
                    total_loss += loss
                    num_batches += 1
                
                test_losses.append(total_loss/num_batches)
                print ('Epoch {} training Loss {:.4f} Accuracy {:.4f}  test Loss {:.4f} Accuracy {:.4f}' .format( \
                                                  epoch + 1, 
                                                  train_losses[-1], 
                                                  train_accuracy.result(),
                                                  test_losses[-1],
                                                  test_accuracy.result()))

                if (test_losses[-1]) < min_test_loss:
                    encoder.save_weights('./saved_weights/Best_model_weights_encoderAttention', save_format='tf')
                    decoder.save_weights('./saved_weights/Best_model_weights_decoderAttention', save_format='tf')
                    min_test_loss = test_loss/test_steps

                train_accuracyVec.append(train_accuracy.result())
                test_accuracyVec.append(test_accuracy.result())
                if epoch % self.predict_every == 0:
                    try:
                        idx = np.random.randint(low=0, high=len(en_test), size=1)[0]
                        predict(en_test[idx], fr_test_out[idx])
                    except:
                        print(" prediction thrown...")

        return (train_losses, test_losses), (train_accuracyVec, test_accuracyVec)
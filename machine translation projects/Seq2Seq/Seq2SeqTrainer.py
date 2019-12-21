import numpy as np
import tensorflow as tf

from Seq2Seqmodel import Encoder, Decoder

class Seq2SeqTrainer:
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

    def predict(self, en_sentence, fr_sentence):
      tokenized_en_sentence = self.en_tokenizer.texts_to_sequences([en_sentence])
      initial_states = self.encoder.init_states(1)
      _, state_h, state_c = self.encoder(tf.constant(tokenized_en_sentence), initial_states, training_mode=False)

      symbol = tf.constant([[self.fr_tokenizer.word_index['<start>']]])
      sentence = []

      while True:
          symbol, state_h, state_c = self.decoder(symbol, (state_h, state_c), training_mode=False)
          # argmax to get max index 
          symbol = tf.argmax(symbol, axis=-1)
          word = self.fr_tokenizer.index_word[symbol.numpy()[0][0]]

          if word == '<end>' or len(sentence) >= len(fr_sentence):
              break

          sentence.append(word)
        
      predicted_sentence = ' '.join(sentence)
      print("--------------PREDICTION--------------")
      print("  English   :  {}" .format(en_sentence))
      print("  Predicted :  {}" .format(predicted_sentence))
      print("  Correct   :  {}" .format(fr_sentence))
      print("------------END PREDICTION------------")

    def train(self, train_dataset_data, test_dataset_data, prediction_data, tokenizers, epochs, restore_checkpoint=False):
        """
            parameters:
                train_dataset_data, test_dataset_data, tokenizers, epochs, restore_checkpoint=True
                
                train_dataset_data should be made from (en_train, fr_train_in, fr_train_out)
                test_dataset_data should be made from (en_test, fr_test_in, fr_test_out)
        """
        
        en_predict, fr_predict = prediction_data
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
        train_dataset_distr = self.strategy.experimental_distribute_dataset(train_dataset)

        test_dataset = tf.data.Dataset.from_tensor_slices((en_test, fr_test_in, fr_test_out))
        test_dataset = test_dataset.shuffle(len(en_test), reshuffle_each_iteration=True)\
                                       .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        test_dataset_distr = self.strategy.experimental_distribute_dataset(test_dataset)
        print("dataset created")
        
        test_losses = []
        train_losses = []
        train_accuracyVec = []
        test_accuracyVec =[]
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        
        prediction_idx = np.random.randint(low=0, high=len(en_predict), size=1)[0]
        prediction_en, prediction_fr = en_predict[prediction_idx], fr_predict[prediction_idx]
        print("input : ", prediction_en)
        print("output: ", prediction_fr)

        with self.strategy.scope():
            self.optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)
            self.encoder = Encoder(en_vocab_size, self.embedding_size, self.lstm_size)
            self.decoder = Decoder(fr_vocab_size, self.embedding_size, self.lstm_size)
            
            ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                       decoder = self.decoder,
                                       optimizer=self.optimizer,
                                       epoch=tf.Variable(1))

            manager = tf.train.CheckpointManager(ckpt, "./checkpoints/Seq2Seq", max_to_keep=5)

            
            if manager.latest_checkpoint and restore_checkpoint:
                ckpt.restore(manager.latest_checkpoint)
                print ('Latest checkpoint restored!!')
            else:
                print("training from scratch")

            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                                     reduction="none") 
            def compute_loss(predictions, labels):
                mask = tf.math.logical_not(tf.math.equal(labels, 0))
                mask = tf.cast(mask, tf.int64)
                per_example_loss = loss_obj(labels, predictions, sample_weight=mask)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

            # one training step
            def train_step(encoder_input, decoder_in, decoder_out, initial_states):
                with tf.GradientTape() as tape:
                    encoder_states = self.encoder(encoder_input, initial_state, training_mode=True)
                    predicted_data, _, _ = self.decoder(decoder_in, encoder_states[1:], training_mode=True)
                    loss = compute_loss(predicted_data, decoder_out)

                trainable = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape.gradient(loss, trainable)
                self.optimizer.apply_gradients(zip(grads, trainable))
                train_accuracy.update_state(decoder_out, predicted_data)
                return loss

            @tf.function
            def distributed_train_step(encoder_input, decoder_in, decoder_out, initial_states):
                per_replica_losses = self.strategy.experimental_run_v2(train_step,
                                                              args=(encoder_input,
                                                                    decoder_in,
                                                                    decoder_out,
                                                                    initial_states,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)
        
            def test_step(encoder_input, decoder_in, decoder_out):
                initial_state = self.encoder.init_states(self.batch_size)
                encoder_states = self.encoder(encoder_input, initial_state, training_mode=False)
                predicted_data, _, _ = self.decoder(decoder_in, encoder_states[1:], training_mode=False)
                loss = compute_loss(predicted_data, decoder_out)

                test_accuracy.update_state(decoder_out, predicted_data)
                return loss

            @tf.function
            def distributed_test_step(encoder_input, decoder_in, decoder_out):
                per_replica_losses = self.strategy.experimental_run_v2(test_step,
                                                              args=(encoder_input,
                                                                    decoder_in,
                                                                    decoder_out,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)
            print("starting training with {} epochs with prediction each {} epoch" .format(epochs, self.predict_every))
            for epoch in range(epochs):
                test_accuracy.reset_states()
                train_accuracy.reset_states()
                initial_state = self.encoder.init_states(self.batch_size)
                total_loss = 0.0
                num_batches = 0
                for _, (en_data, fr_data_in, fr_data_out) in enumerate(train_dataset_distr):
                    loss = distributed_train_step(en_data, fr_data_in, fr_data_out, initial_state)
                    total_loss += loss
                    num_batches += 1
                train_losses.append(total_loss/num_batches)
                total_loss = 0.0
                num_batches = 0
                for _, (en_data, fr_data_in, fr_data_out) in enumerate(test_dataset_distr):
                    loss = distributed_test_step(en_data, fr_data_in, fr_data_out)
                    total_loss += loss
                    num_batches += 1
                test_losses.append(total_loss/num_batches)
                print ('Epoch {} training Loss {:.4f} Accuracy {:.4f}  test Loss {:.4f} Accuracy {:.4f}' .format(
                                                      epoch + 1, 
                                                      train_losses[-1], 
                                                      train_accuracy.result(),
                                                      test_losses[-1],
                                                      test_accuracy.result()))
                train_accuracyVec.append(train_accuracy.result())
                test_accuracyVec.append(test_accuracy.result())
                ckpt.epoch.assign_add(1)
                if int(epoch) % 5 == 0:
                    save_path = manager.save()
                    print("Saving checkpoint for epoch {}: {}".format(epoch, save_path))

                if epoch % self.predict_every == 0:
                    self.predict(prediction_en, prediction_fr)
        return (train_losses, test_losses), (train_accuracyVec, test_accuracyVec)

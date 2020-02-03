import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from TrainerBase import BaseTrainer

sys.path.insert(0, r"../models")
sys.path.insert(0, r"../utilities")

from Seq2SeqAttmodel import Encoder, Decoder
from utils import makeDatasets, save_to_csv

class Seq2SeqAttentionTrainer(BaseTrainer):
    def __init__(self, batch_size, lstm_size, embedding_size, tokenizers, predict_every):
        """
            Parameters: 
                batch_size - batch_size of input data,
                lstm_size - number of lstm units
                embedding_size - embedding size for wholde model    
                tokenizers - two tokenizers for input and output data. Should be in form en_tokenizer, fr_tokenizer
                predict_every - how often to write prediction during training
        """
        
        self.en_tokenizer, self.fr_tokenizer = tokenizers
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.predict_every = predict_every
        self.strategy = tf.distribute.MirroredStrategy()
        self.encoder = None
        self.decoder = None
        self.optimizer = None

    def translate(self, en_sentence):
        """
            Translates sentence
            
            Parameters:
                en_sentence - sentence that will be translated
            
            Returns:
                translated sentence
                alignments matrix - attention matrix for given translation
        """
        tokenized_input_data = self.en_tokenizer.encode(en_sentence)      
        tokenized_input_data = tf.expand_dims(tokenized_input_data, 0)
        initial_states = self.encoder.init_states(1)
        encoder_out, state_h, state_c = self.encoder(tf.constant(tokenized_input_data), 
                                                     initial_states, 
                                                     training_mode=False)

        decoder_in = [self.fr_tokenizer.vocab_size]
        decoder_in = tf.expand_dims(decoder_in, 0)
        end_tag = self.fr_tokenizer.vocab_size+1
        output_seq = []
        alignments = []
        while True:
            decoder_out, state_h, state_c, alignment = self.decoder(decoder_in, 
                                                                    (state_h, state_c), 
                                                                    encoder_out, 
                                                                    training_mode=False)
            # argmax to get max index 
            decoder_in = tf.expand_dims(tf.argmax(decoder_out, -1), 0)
            predicted_data = decoder_in

            if  predicted_data.numpy()[0] == end_tag or len(output_seq) >=40:
                break
             
            alignments.append(alignment)
            output_seq.append(self.fr_tokenizer.decode(predicted_data.numpy()[0]))
            
        return "".join(output_seq), alignments

    def train(self, train_data, test_data, prediction_data, epochs, attention_type="general", restore_checkpoint=False, csv_name="seq2seqAttention_data.csv"):
        """
            Training method that uses distributed training
            
            Parameters:
                train_data - input data for training. Should be in form : en_train, fr_train_in, fr_train_out
                test_data - input data for test step. Should be in form : en_test, fr_test_in, fr_test_out
                prediction_data - input data for prediction step. Should be in form of: en_predict, fr_predict
                epochs - number of epochs that should be run
                attention_type - what attention method to use " dot/general/concat. Default - general
                restore_checkpoint - should we restore last checkpoint and resume training. Defualt set to false.
                csv_name - name of csv file where losses/accuracies will be saved. default = seq2seqAttention_data.csv.
                  If restore_checkpoint is set to False, file will be erased and only current run will be present.
            
            Returns:
                tuple losses, accuracy where losses = (train_losses, test_losses), accuracy = (train-accuracy, test_accuracy)
        """
        
        print_heatmap=True
        
        en_predict, fr_predict = prediction_data
        en_vocab_size = self.en_tokenizer.vocab_size
        fr_vocab_size = self.fr_tokenizer.vocab_size + 2
        
        print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
        GLOBAL_BATCH_SIZE = self.batch_size*self.strategy.num_replicas_in_sync

        train_dataset_distr, test_dataset_distr = makeDatasets(train_data, test_data, GLOBAL_BATCH_SIZE, self.strategy)
        
        test_losses = []
        train_losses = []
        train_accuracyVec = []
        test_accuracyVec =[]
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        prediction_idx = np.random.randint(low=0, high=len(en_predict), size=1)[0]
        prediction_en, prediction_fr = en_predict[prediction_idx], fr_predict[prediction_idx]
        print("prediction input : ", prediction_en)
        print("prediction output: ", prediction_fr)

        if not os.path.exists("heatmap"):
          os.mkdir("heatmap")

        alignments = []

        with self.strategy.scope():
            self.encoder = Encoder(lstm_size=self.lstm_size, 
                                   embedding_size=self.embedding_size, 
                                   vocab_size=en_vocab_size)
            
            self.decoder = Decoder(lstm_size=self.lstm_size, 
                                   embedding_size=self.embedding_size, 
                                   vocab_size=fr_vocab_size, 
                                   attention_type=attention_type)
            
            self.optimizer = tf.keras.optimizers.Adam(clipnorm=0.5)
            
            ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                       decoder = self.decoder,
                                       optimizer=self.optimizer,
                                       epoch=tf.Variable(1))

            manager = tf.train.CheckpointManager(ckpt, "./checkpoints/Seq2SeqAttention", max_to_keep=5)

            
            if manager.latest_checkpoint and restore_checkpoint:
                ckpt.restore(manager.latest_checkpoint)
                print ('Latest checkpoint restored!!')
            else:
                print("training from scratch")

            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True, reduction="none")
            
            def compute_loss(predictions, labels):
                mask = tf.math.logical_not(tf.math.equal(labels, 0))
                mask = tf.cast(mask, tf.int64)
                per_example_loss = loss_obj(labels, predictions, sample_weight=mask)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
            
            # one training step
            def train_step(en_data, fr_data_in, fr_data_out, initial_states):
                loss = 0
                predicted_output = None
                train_accuracy.reset_states()
                with tf.GradientTape() as tape:
                    encoder_output, state_h, state_c = self.encoder(en_data, 
                                                                    initial_states, 
                                                                    training_mode=True)
                    # shape[1] because we want each word for all batches
                    for i in range(fr_data_out.shape[1]):
                        decoder_input = tf.expand_dims(fr_data_in[:,i], 1)
                        decoder_output, state_h, state_c, _ = self.decoder(decoder_input,
                                                                           (state_h, state_c),
                                                                           encoder_output,
                                                                           training_mode=True)
                        
                        loss +=compute_loss(decoder_output, fr_data_out[:,i])
                        decoder_output = tf.expand_dims(decoder_output, axis=1)
                        if i == 0:
                          predicted_output = decoder_output
                        else:
                          predicted_output = tf.concat([predicted_output, decoder_output], axis=1)

                trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients(zip(grads, trainable_vars))

                train_accuracy.update_state(fr_data_out, predicted_output)

                return loss / fr_data_out.shape[1]

            @tf.function
            def distributed_train_step(en_data, fr_data_in, fr_data_out, initial_states):
                per_replica_losses = self.strategy.experimental_run_v2(train_step,
                                                                  args=(en_data,
                                                                        fr_data_in,
                                                                        fr_data_out,
                                                                        initial_states,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            def test_step(en_data, fr_data_in, fr_data_out):
                loss = 0
                predicted_output = []
                initial_states = self.encoder.init_states(self.batch_size)
                encoder_output, state_h, state_c = self.encoder(en_data, 
                                                                initial_states, 
                                                                training_mode=False)
                for i in range(fr_data_out.shape[1]):
                    decoder_input = tf.expand_dims(fr_data_in[:,i], 1)
                    decoder_output, state_h, state_c, _ = self.decoder(decoder_input,
                                                                    (state_h, state_c),
                                                                    encoder_output,
                                                                    training_mode=False)
                    loss +=compute_loss(decoder_output, fr_data_out[:,i])

                    decoder_output = tf.expand_dims(decoder_output, axis=1)
                    if i == 0:
                      predicted_output = decoder_output
                    else:
                      predicted_output = tf.concat([predicted_output, decoder_output], axis=1)
                      
                test_accuracy.update_state(fr_data_out, predicted_output)

                return loss/fr_data_out.shape[1]

            @tf.function
            def distributed_test_step(en_data, fr_data_in, fr_data_out):
                per_replica_losses = self.strategy.experimental_run_v2(test_step,
                                                                 args=(en_data,
                                                                       fr_data_in,
                                                                       fr_data_out,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            print("starting training with {} epochs with prediction each {} epoch" .format(epochs, self.predict_every))
            for epoch in range(epochs):
                test_accuracy.reset_states()
                train_accuracy.reset_states()
                initial_states = self.encoder.init_states(self.batch_size)
                
                total_loss = 0.0
                num_batches = 0
                for _, (en_data, fr_data_in, fr_data_out) in enumerate(train_dataset_distr):
                    loss = distributed_train_step(en_data, fr_data_in, fr_data_out, initial_states)
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
                print ('Epoch {} training Loss {:.4f} Accuracy {:.4f}  test Loss {:.4f} Accuracy {:.4f}' .format( \
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

                predicted, alignment  = self.translate(prediction_en)
                
                if epoch % self.predict_every == 0:
                    print("----------------------------PREDICTION----------------------------")
                    print("Predicted:  {} " .format(predicted))
                    print("Should be:  {} " .format(prediction_fr))
                    print("--------------------------END PREDICTION--------------------------")
                    
                if print_heatmap:
                    attention_map = np.squeeze(alignment, (1, 2))
                    alignments.append(attention_map)
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.matshow(attention_map, cmap='jet')
                    ax.set_xticklabels([''] + prediction_en.split(' '), rotation=90)
                    ax.set_yticklabels([''] + predicted.split(' '))

                    plt.savefig('heatmap/prediction_{}.png' .format(epoch))
                    #plt.show()
                    plt.close()
                    
        save_path = manager.save()
        print ('Saving checkpoint for end at {}'.format(save_path))
        save_to_csv(losses=(train_losses, test_losses), 
                    accuracy=(train_accuracyVec, test_accuracyVec), 
                    append=restore_checkpoint,
                    file_name=csv_name)

        return (train_losses, test_losses), (train_accuracyVec, test_accuracyVec)
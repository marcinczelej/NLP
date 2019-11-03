import tensorflow as tf
import numpy as np

from params import *
from model import *

def train(en_data, fr_data, en_test, fr_test, vocab_data, fr_tokenizer, en_tokenizer, attention_type):
    fr_train_in, fr_train_out, fr_test_tokenized = fr_data
    en_train, en_test_tokenized = en_data
    en_vocab_size, fr_vocab_size = vocab_data
    
    train_dataset = tf.data.Dataset.from_tensor_slices((en_train, fr_train_in, fr_train_out))
    train_dataset = train_dataset.shuffle(len(en_train), reshuffle_each_iteration=True)\
                                 .batch(BATCH_SIZE, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((en_test_tokenized, fr_test, fr_test_tokenized))
    test_dataset = test_dataset.shuffle(len(en_test_tokenized), reshuffle_each_iteration=True)\
                               .batch(BATCH_SIZE, drop_remainder=True)

    # lost function with zeros masked
    def loss_fn(real, targets):
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      mask = tf.math.logical_not(tf.math.equal(targets, 0))
      mask = tf.cast(mask, tf.int64)

      return loss(targets, real, sample_weight=mask)
    
    min_test_loss = 10000.0
    test_losses = []
    train_losses = []
    test_loss = tf.keras.metrics.Mean()
    training_loss = tf.keras.metrics.Mean()
    optim = tf.keras.optimizers.Adam(clipnorm=0.5)

    encoder = Encoder(LSTM_SIZE, EMBEDDING_SIZE, en_vocab_size)
    decoder = Decoder(LSTM_SIZE, EMBEDDING_SIZE, fr_vocab_size, attention_type)

    def predict_output(en_sentence, fr_sentence):
        should_be_sentence = fr_sentence
        sentence = en_tokenizer.texts_to_sequences([en_sentence])

        initial_states = encoder.init_states(1)
        encoder_out, state_h, state_c = encoder(tf.constant(sentence), initial_states, training=False)

        decoder_in = tf.constant([[fr_tokenizer.word_index['<start>']]])
        sentence = []

        while True:
            decoder_out, state_h, state_c = decoder( \
                decoder_in, (state_h, state_c), encoder_out, training=False)
                        # argmax to get max index 
            decoder_in = tf.expand_dims(tf.argmax(decoder_out, -1), 0)
                        #print("word = ", decoder_in.numpy()[0][0])
            word = fr_tokenizer.index_word[decoder_in.numpy()[0][0]]

            if  word == '<end>':
                break
            
            sentence.append(word)
        predicted_sentence = ' '.join(sentence)
        print("----------------------------PREDICTION----------------------------")
        print("       En sentence {} " .format(en_sentence))
        print("       Predicted:  {} " .format(predicted_sentence))
        print("       Should be:  {} " .format(should_be_sentence))
        print("--------------------------END PREDICTION--------------------------")
    
    @tf.function
    def test_step(en_data, fr_data_tokenized):
        loss =0
        initial_states = encoder.init_states(BATCH_SIZE)
        encoder_output, state_h, state_c = encoder(en_data, initial_states, training=False)
        decoder_input = tf.constant(fr_tokenizer.word_index['<start>'], shape=(BATCH_SIZE, 1))
        
        for i in range(fr_data_tokenized.shape[1]):
            decoder_output, state_h, state_c = decoder(
                decoder_input, (state_h, state_c), encoder_output, training=False)
            decoder_input =tf.expand_dims(tf.argmax(decoder_output, 1),1)
            loss +=loss_fn(decoder_output, fr_data_tokenized[:,i])

        return loss/fr_data_tokenized.shape[1]
    
    # en_data - english sentences to be translated [batch_size, en_max_len]
    # fr_data_in - input data to be sent ot decoder [batch_size, fr_max_len]
    # fr_data_out - franch data to be taken while calculation loss
    # initial_states - initial states of encoder
    #
    # because attention is per word not per seq we have to decode every word separated
    @tf.function
    def train_step(en_data, fr_data_in, fr_data_out, initial_states):
      loss = 0
      with tf.GradientTape() as tape:
        encoder_output, state_h, state_c = encoder(en_data, initial_states)
        # shape[1] because we want each word for all batches
        for i in range(fr_data_out.shape[1]):
          decoder_input = tf.expand_dims(fr_data_in[:,i], 1)
          decoder_output, state_h, state_c = decoder(decoder_input, (state_h, state_c), encoder_output)
          loss +=loss_fn(decoder_output, fr_data_out[:,i])

      trainable_vars = encoder.trainable_variables + decoder.trainable_variables
      grads = tape.gradient(loss, trainable_vars)
      optim.apply_gradients(zip(grads, trainable_vars))

      return loss / fr_data_out.shape[1]
    
    print("batches each epoch : ", len(en_train)/BATCH_SIZE)
    print(" batches per epoch: ", len(fr_train_in)//BATCH_SIZE)

    for epoch in range(EPOCHS):
        initial_states = encoder.init_states(BATCH_SIZE)

        for batch, (en_data, fr_data_in, fr_data_out) in enumerate(train_dataset.take(-1)):
            loss = train_step(en_data, fr_data_in, fr_data_out, initial_states)
            training_loss.update_state(loss)
        
        for batch, (en_data, fr_data, fr_data_tokenized) in enumerate(test_dataset.take(-1)):
            loss = test_step(en_data, fr_data_tokenized)
            test_loss.update_state(loss)

        train_losses.append(training_loss.result().numpy())
        test_losses.append(test_loss.result().numpy())
        print("Epoch {} : train loss {} test loss : {}" \
                  .format(epoch, train_losses[-1], test_losses[-1]))
        
        if epoch%10 == 0:
                idx = np.random.choice(len(en_test))
                predict_output(en_test[idx], fr_test[idx])
        if (test_losses[-1]) < min_test_loss:
            encoder.save_weights('./saved_weights/Best_model_weights_encoder', save_format='tf')
            decoder.save_weights('./saved_weights/Best_model_weights_decoder', save_format='tf')
            min_test_loss = test_losses[-1]
    return train_losses, test_losses
            
def distributedTrain(en_data, fr_data, en_test, fr_test, vocab_data, fr_tokenizer, en_tokenizer, attention_type):
    fr_train_in, fr_train_out, fr_test_tokenized = fr_data
    en_train, en_test_tokenized = en_data
    en_vocab_size, fr_vocab_size = vocab_data

    strategy = tf.distribute.MirroredStrategy()
    GLOBAL_BATCH_SIZE = BATCH_SIZE*strategy.num_replicas_in_sync
    train_steps = len(en_train)//GLOBAL_BATCH_SIZE
    test_steps = len(en_test_tokenized)//GLOBAL_BATCH_SIZE
    print("Number of epochs : {}" .format(EPOCHS))
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    print("GLOBAL_BATCH_SIZE : {}" .format(GLOBAL_BATCH_SIZE))
    print("train batches each epoch : ", train_steps)
    print("test batches each epoch : ", test_steps)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((en_train, fr_train_in, fr_train_out))
    train_dataset = train_dataset.shuffle(len(en_train), reshuffle_each_iteration=True) \
                                 .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((en_test_tokenized, fr_test, fr_test_tokenized))
    test_dataset = test_dataset.shuffle(len(en_test_tokenized), reshuffle_each_iteration=True) \
                               .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    train_losses = []
    test_losses = []
    min_test_loss = 10000.0

    with strategy.scope():
        encoder = Encoder(LSTM_SIZE, EMBEDDING_SIZE, en_vocab_size)
        decoder = Decoder(LSTM_SIZE, EMBEDDING_SIZE, fr_vocab_size, attention_type)
        optim = tf.keras.optimizers.Adam(clipnorm=0.5)
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        encoder.save_weights('./saved_weights/starting_model_encoder', save_format='tf')
        decoder.save_weights('./saved_weights/starting_model_decoder', save_format='tf')
        
        # should get en_sentence after texts_to_sequences([...])
        def predict_output(en_sentence, fr_sentence):
            should_be_sentence = fr_sentence
            sentence = en_tokenizer.texts_to_sequences([en_sentence])
            
            initial_states = encoder.init_states(1)
            encoder_out, state_h, state_c = encoder(tf.constant(sentence), initial_states, training=False)

            decoder_in = tf.constant([[fr_tokenizer.word_index['<start>']]])
            sentence = []
            while True:
                decoder_out, state_h, state_c = decoder( \
                             decoder_in, (state_h, state_c), encoder_out, training=False)
                # argmax to get max index 
                decoder_in = tf.expand_dims(tf.argmax(decoder_out, -1), 0)
                word = fr_tokenizer.index_word[decoder_in.numpy()[0][0]]

                if  word == '<end>':
                    break
                sentence.append(word)

            predicted_sentence = ' '.join(sentence)
            print("----------------------------PREDICTION----------------------------")
            print("       En sentence {} " .format(en_sentence))
            print("       Predicted:  {} " .format(predicted_sentence))
            print("       Should be:  {} " .format(should_be_sentence))
            print("--------------------------END PREDICTION--------------------------")

        def compute_loss(predictions, labels):
            mask = tf.math.logical_not(tf.math.equal(labels, 0))
            mask = tf.cast(mask, tf.int64)
            per_example_loss = loss_obj(labels, predictions, sample_weight=mask)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        # one training step
        def train_step(en_data, fr_data_in, fr_data_out, initial_states):
          loss = 0
          with tf.GradientTape() as tape:
            encoder_output, state_h, state_c = encoder(en_data, initial_states, training=True)
            # shape[1] because we want each word for all batches
            for i in range(fr_data_out.shape[1]):
              decoder_input = tf.expand_dims(fr_data_in[:,i], 1)
              decoder_output, state_h, state_c = decoder(
                  decoder_input, (state_h, state_c), encoder_output, training=True)
              loss +=compute_loss(decoder_output, fr_data_out[:,i])

          trainable_vars = encoder.trainable_variables + decoder.trainable_variables
          grads = tape.gradient(loss, trainable_vars)
          optim.apply_gradients(zip(grads, trainable_vars))

          return loss / fr_data_out.shape[1]

        @tf.function
        def distributed_train_step(en_data, fr_data_in, fr_data_out, initial_states):
            per_replica_losses = strategy.experimental_run_v2(train_step,
                                                              args=(en_data,
                                                                    fr_data_in,
                                                                    fr_data_out,
                                                                    initial_states,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        def test_step(en_data, fr_data_tokenized):
            loss =0
            initial_states = encoder.init_states(BATCH_SIZE)
            encoder_output, state_h, state_c = encoder(en_data, initial_states, training=False)

            decoder_input = tf.constant(fr_tokenizer.word_index['<start>'], shape=(BATCH_SIZE, 1))

            for i in range(fr_data_tokenized.shape[1]): 
                decoder_output, state_h, state_c = decoder(
                    decoder_input, (state_h, state_c), encoder_output, training=False)
                decoder_input =tf.expand_dims(tf.argmax(decoder_output, 1),1)
                loss +=compute_loss(decoder_output, fr_data_tokenized[:,i])
            return loss/fr_data_tokenized.shape[1]

        @tf.function
        def distributed_test_step(en_data, fr_data_tokenized):
            per_replica_losses = strategy.experimental_run_v2(test_step,
                                                             args=(en_data,
                                                                   fr_data_tokenized,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        for epoch in range(EPOCHS):
            test_loss = 0.0
            training_loss = 0.0
            initial_states = encoder.init_states(BATCH_SIZE)

            for batch, (en_data, fr_data_in, fr_data_out) in enumerate(train_dist_dataset):
                loss = distributed_train_step(en_data, fr_data_in, fr_data_out, initial_states)
                training_loss+=loss
            for batch, (en_data, fr_data, fr_data_tokenized) in enumerate(test_dist_dataset):
                loss = distributed_test_step(en_data, fr_data_tokenized)
                test_loss+=loss

            print("Epoch {} : train loss {} test loss : {}" \
                  .format(epoch, training_loss/train_steps, test_loss/test_steps))
            train_losses.append(training_loss/train_steps)
            test_losses.append(test_loss/test_steps)

            if (test_loss/test_steps) < min_test_loss:
                encoder.save_weights('./saved_weights/Best_model_weights_encoder', save_format='tf')
                decoder.save_weights('./saved_weights/Best_model_weights_decoder', save_format='tf')
                min_test_loss = test_loss/test_steps
            

            idx = np.random.choice(len(en_test))
            predict_output(en_test[idx], fr_test[idx])

        return train_losses, test_losses

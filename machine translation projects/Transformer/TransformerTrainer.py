import numpy as np
import tensorflow as tf

from model import *

class customLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
  """
  according to Attention is all you need paper learning rate has custom scheduler:
  there are two parameters : 
  - d_model
  - warmup_steps ( in paper set to 4000)
  according to paper https://arxiv.org/pdf/1706.03762.pdf
  point 5.3 Optimizer
  """
  def __init__(self, warmup_steps, d_model):
    super(customLearningRate, self).__init__()
    self.d_model = tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps
  
  def __call__(self, step):
    firstScheduler = tf.math.rsqrt(step)
    secondScheduler = step*(self.warmup_steps**-1.5)
    return tf.math.rsqrt(self.d_model)*tf.math.minimum(firstScheduler, secondScheduler)

class TransformerTrainer:
    def __init__(self, batch_size, num_layers, d_model, dff, num_heads, predict_every):
        """
            Parameters: 
                batch_size, num_layers, d_model, dff, nim_heads, predict_every
        """
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.dff = dff
        self.num_heads = num_heads
        self.predict_every = predict_every
        self.strategy = tf.distribute.MirroredStrategy()
        self.transformer_model = None
        self.optimizer = None
        self.fr_tokenizer = None
        self.en_tokenizer = None
        self.checkpoint_path = "./checkpoints/train"

    def train(self, train_dataset_data, test_dataset_data, tokenizers, epochs, restore_checkpoint=False):
        """
            Parameters:
                train_dataset_data, test_dataset_data, tokenizers, epochs, restore_checkpoint=True
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
        train_dataset_distr = self.strategy.experimental_distribute_dataset(train_dataset)

        test_dataset = tf.data.Dataset.from_tensor_slices((en_test, fr_test_in, fr_test_out))
        test_dataset = test_dataset.shuffle(len(en_test), reshuffle_each_iteration=True)\
                                       .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        test_dataset_distr = self.strategy.experimental_distribute_dataset(test_dataset)

        # distributed train

        test_losses = []
        train_losses = []
        train_accuracyVec = []
        test_accuracyVec =[]
        test_loss = tf.keras.metrics.Mean()
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        with self.strategy.scope():
          custom_learning_rate = customLearningRate(warmup_steps=4000,
                                                    d_model=self.d_model)

          self.optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate,
                                              beta_1=0.9,
                                              beta_2=0.98,
                                              epsilon=1e-9)

          self.transformer_model = Transformer(embedding_size=self.d_model,
                                          dff=self.dff,
                                          input_max_seq_length=2000,
                                          output_max_seq_length=1855,
                                          input_vocab_size=en_vocab_size,
                                          output_vocab_size=fr_vocab_size,
                                          encoder_blocks=self.num_layers,
                                          decoder_blocks=self.num_layers,
                                          heads=self.num_heads)

          ckpt = tf.train.Checkpoint(transformer=self.transformer_model,
                                    optimizer=self.optimizer,
                                    epoch=tf.Variable(1))

          manager = tf.train.CheckpointManager(ckpt, self.checkpoint_path, max_to_keep=5)


          if manager.latest_checkpoint and restore_checkpoint:
              ckpt.restore(manager.latest_checkpoint)
              print ('Latest checkpoint restored!!')
          else:
              print("training from scratch")

          loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True, reduction="none")

          def loss_fn(real, targets):
              mask = tf.math.logical_not(tf.math.equal(targets, 0))
              mask = tf.cast(mask, tf.int64)
              per_example_loss = loss_object(targets, real, sample_weight=mask)
              return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)  

          def train_step(input_data, real_data_in, real_data_out):
              encoder_pad_mask = makePaddingMask(input_data)
              elements_mask = makeSequenceMask(real_data_in.shape[1])
              with tf.GradientTape() as tape:
                predicted_data = self.transformer_model(
                                                input_data,
                                                real_data_in,
                                                encoder_pad_mask,
                                                elements_mask,
                                                training_enabled=True,
                                                training=True)
                loss = loss_fn(predicted_data, real_data_out)

              trainable_vars = self.transformer_model.trainable_variables
              grads = tape.gradient(loss, trainable_vars)
              self.optimizer.apply_gradients(zip(grads, trainable_vars))
              train_accuracy.update_state(real_data_out, predicted_data)
              return loss

          @tf.function
          def distributed_train_step(input_data, real_data_in, real_data_out):
              per_replica_losses = self.strategy.experimental_run_v2(train_step,
                                                              args=(input_data,
                                                                    real_data_in,
                                                                    real_data_out))
              return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


          def test_step(input_data, real_data_in, real_data_out):
              encoder_pad_mask = makePaddingMask(input_data)
              elements_mask = makeSequenceMask(real_data_in.shape[1])
              predicted_data = self.transformer_model(
                                                  input_data,
                                                  real_data_in,
                                                  encoder_pad_mask,
                                                  elements_mask,
                                                  training_enabled=False,
                                                  training=False)
              loss = loss_fn(predicted_data, real_data_out)

              test_accuracy.update_state(real_data_out, predicted_data)
              return loss

          @tf.function
          def distributed_test_step(input_data, real_data_in, real_data_out):
              per_replica_losses = self.strategy.experimental_run_v2(test_step, args=(input_data,
                                                          real_data_in,
                                                          real_data_out,))
              return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

          for epoch in range(epochs):
              total_loss = 0
              num_batches = 0
              test_loss.reset_states()
              test_accuracy.reset_states()
              train_accuracy.reset_states()

              for _, (en_data, fr_data_in, fr_train_out) in enumerate(train_dataset_distr):
                  loss = distributed_train_step(en_data, fr_data_in, fr_train_out)
                  total_loss += loss
                  num_batches += 1
              train_losses.append(total_loss/num_batches)

              total_loss = 0
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

              idx = np.random.randint(low=0, high=len(en_test), size=1)[0]
              self.predict(en_test[idx], fr_test_out[idx])

              ckpt.epoch.assign_add(1)
              if int(epoch) % 5 == 0:
                  save_path = manager.save()
                  print("Saving checkpoint for epoch {}: {}".format(epoch, save_path))

          save_path = manager.save()
          print ('Saving checkpoint for end at {}'.format(save_path))
            
          return (train_losses, test_losses), (train_accuracyVec, test_accuracyVec)

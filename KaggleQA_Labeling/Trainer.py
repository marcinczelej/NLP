import os

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
import horovod.tensorflow as hvd

from parameters import *
from pseudoLabeler import PseudoLabeler
from utilities import accumulated_gradients ,CustomSchedule
from metric import *

class Trainer(object):

    @classmethod
    def train(cls, model, tokenizer, preprocessedInput, targets, preprocessedPseudo):

        kf = KFold(n_splits=5)
        fold_nr =0

        for train_idx, test_idx in kf.split(preprocessedInput):

            # train test indices
            train_input = tf.gather(preprocessedInput, train_idx, axis=0)
            train_target = tf.gather(targets, train_idx, axis=0)

            test_input = tf.gather(preprocessedInput, test_idx, axis=0)
            test_target = tf.gather(targets, test_idx, axis=0)

            #train dataset
            train_ds = tf.data.Dataset.from_tensor_slices((train_input, train_target)). \
                                     shuffle(len(train_input)//4, reshuffle_each_iteration=True). \
                                     batch(batch_size=batch_size, drop_remainder=True)

            #test dataset
            test_ds = tf.data.Dataset.from_tensor_slices((test_input, test_target)). \
                                     shuffle(len(test_input)//4, reshuffle_each_iteration=True). \
                                     batch(batch_size=batch_size, drop_remainder=True)

            lr_scheduler = CustomSchedule(warmup_steps=warmup_steps*2, 
                                          num_steps=targets.shape[0]//batch_size, 
                                          base_lr=lr*hvd.size())

            #optimizer = tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=decay)
            optimizer = tf.optimizers.Adam(learning_rate = lr_scheduler)
            bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            """model = BertForQALabeling.from_pretrained('bert-base-uncased', num_labels=num_labels, output_hidden_states=True)"""

            """model = RoBERTaForQALabeling.from_pretrained('roberta-base', num_labels=num_labels, output_hidden_states=True)"""

            """model = RoBERTaForQALabelingMultipleHeads.from_pretrained('roberta-base', num_labels=num_labels, output_hidden_states=True)"""

            trainable = model.trainable_variables

            checkpoint_dir = './checkpoints/best_{}_fold_{}' .format(model.getName(), fold_nr)
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)


            @tf.function
            def train_step(inputs, y_true, first_batch):
                with tf.GradientTape() as tape:
                    ids_mask, type_ids_mask, attention_mask = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]
                    y_pred = model(ids_mask, 
                                     attention_mask= attention_mask, 
                                     token_type_ids=type_ids_mask, 
                                     training=True)
                    loss = tf.reduce_sum(bce_loss(y_true, y_pred)*(1. / batch_size))

                tape = hvd.DistributedGradientTape(tape)

                grads = tape.gradient(loss, trainable)

                return loss, grads, tf.math.sigmoid(y_pred)

            @tf.function
            def test_step(inputs, y_true):
                ids_mask, type_ids_mask, attention_mask = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]
                y_pred = model(ids_mask, 
                              attention_mask= attention_mask, 
                              token_type_ids=type_ids_mask, 
                              training=False)
                loss = tf.reduce_sum(bce_loss(y_true, y_pred)*(1. / batch_size))

                return loss, tf.math.sigmoid(y_pred)

            last_loss = 999999

            print("starting training for {} epochs" .format(epochs))
            for epoch in range(epochs):
                gradients = None
                train_losses = []
                test_losses = []
                train_preds = []
                test_preds = []
                train_targets = []
                test_targets = []
                global_batch = 0
                for batch_nr, (inputs, y_true) in enumerate(train_ds):
                    loss, current_gradient, y_pred = train_step(inputs, y_true, batch_nr==0)
                    train_losses.append(np.mean(loss))
                    train_preds.append(y_pred)
                    train_targets.append(y_true)
                    gradients = accumulated_gradients(gradients, current_gradient, gradient_accumulate_steps)

                    if (batch_nr +1)%gradient_accumulate_steps ==0:
                        #print("batch_nr {} gradient applying" .format(batch_nr))
                        optimizer.apply_gradients(zip(gradients, trainable))
                        global_batch +=1
                        gradients = None

                        if batch_nr == 0:
                            print("first batch")
                            hvd.broadcast_variables(trainable, root_rank=0)
                            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

                    """if batch_nr % 100 == 0 and hvd.local_rank() == 0:
                        print('Step {} loss {}'  .format(batch_nr, loss, ))"""

                for _, (inputs, y_true) in enumerate(test_ds):
                    loss, y_pred = test_step(inputs, y_true)
                    test_losses.append(np.mean(loss))
                    test_preds.append(y_pred)
                    test_targets.append(y_true)

                test_spearmans = spearman_metric(np.vstack(test_targets), np.vstack(test_preds))
                train_spearmans = spearman_metric(np.vstack(train_targets), np.vstack(train_preds))
                
                print("type test_spearman ", type(test_spearmans))
                print("test spearman ", test_spearmans)

                print("epoch {} train loss {} test loss {} test spearman metric {} train spearman metric {}" \
                      .format(epoch, np.mean(train_losses), np.mean(test_losses), test_spearmans, train_spearmans))

                if np.mean(test_spearmans) < last_loss:
                    if hvd.rank() == 0:
                        checkpoint.save(os.path.join(checkpoint_dir, "best_{}_best_model" .format(model.getName())))
                        last_loss = np.mean(test_spearmans)
                        print("saving checkpoint for {}... " .format(model.getName()))

            """    
                Pseudo labeling for given fold using stackexchange data

                    1. restoring best checkpoint for given fold
                    2. predicting output values for stackexchange
                    3. saving predicted data into csv file 
            """
            pseudo_labeling_ds = tf.data.Dataset.from_tensor_slices((preprocessedPseudo)).batch(batch_size=batch_size, drop_remainder=True)
            
            PseudoLabeler.create_pseudo_labels(checkpoint=checkpoint, 
                                               model=model,
                                               optimizer=optimizer, 
                                               checkpoint_dir=checkpoint_dir, 
                                               pseudo_labeling_df=pseudo_labeling_df, 
                                               fold_nr=fold_nr)
            fold_nr += 1
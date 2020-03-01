import os
import numpy as np

import pandas as pd
import tensorflow as tf

from transformers import BertTokenizer
from model import BertForQALabeling
from preprocessing import dataPreprocessor
from parameters import *

from sklearn.model_selection import KFold

import horovod.tensorflow as hvd

from scipy.stats import spearmanr

hvd.init()

gpus = tf.config.list_physical_devices('GPU') 
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    print("gpus ", gpus)
    print("local rank ",hvd.local_rank())
    tf.config.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    print(tf.config.get_visible_devices())

data_dir = "google-quest-challenge/"

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
submit_df = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))    

train_X_df = train_df[['question_title', 'question_body', 'answer', 'category']]
train_targets_df = train_df[["question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written"]]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

q_title = train_X_df['question_title'].values
q_body = train_X_df['question_body'].values
answer = train_X_df['answer'].values

targets = train_targets_df.to_numpy()

dataPreprocessor.logger = False
dataPreprocessor.tokenizer = tokenizer

preprocessedInput = dataPreprocessor.preprocessBatch(q_body, q_title, answer, max_seq_lengths=(30,128,128, 290))

train_ds = tf.data.Dataset.from_tensor_slices((preprocessedInput, targets)).shuffle(len(q_body)//4, reshuffle_each_iteration=True).batch(batch_size=batch_size, drop_remainder=True)
print("Batches ", targets.shape[0]//batch_size)

print_each = targets.shape[0]//batch_size//5

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps, num_steps, base_lr):
    super(CustomSchedule, self).__init__()

    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    self.num_steps = tf.cast(num_steps, tf.float32)
    self.lr = tf.cast(base_lr, tf.float32)

  def __call__(self, step):
    def warmupPhase() : return step/tf.math.maximum(1.0, self.warmup_steps)
    def decayPhase() : return tf.math.maximum(0.0, (self.num_steps - step))/tf.math.maximum(1.0, self.num_steps - self.warmup_steps)

    multiplier = tf.cond(tf.math.less(step, self.warmup_steps), warmupPhase, decayPhase)
    
    return self.lr * multiplier

def accumulated_gradients(gradients,
                          step_gradients,
                          num_grad_accumulates) -> tf.Tensor:
    if gradients is None:
        gradients = [flat_gradients(g) / num_grad_accumulates for g in step_gradients]
    else:
        for i, g in enumerate(step_gradients):
            gradients[i] += flat_gradients(g) / num_grad_accumulates
    
    return gradients

# This is needed for tf.gather like operations.
def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    '''Convert gradients if it's tf.IndexedSlices.
    When computing gradients for operation concerning `tf.gather`, the type of gradients 
    '''
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices

def spearman_metric(y_true, y_pred):
    corr = [
        spearmanr(pred_col, target_col).correlation
        for pred_col, target_col in zip(y_pred.T, y_true.T)
    ]
    return corr

kf = KFold(n_splits=5)
fold_nr =0

for train_idx, test_idx in kf.split(preprocessedInput):
    print("                FOLD ", fold_nr)
    
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
    optimizer = tf.optimizers.Adam(learning_rate = lr)
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    model = BertForQALabeling.from_pretrained('bert-base-uncased', num_labels=num_labels, output_hidden_states=True)
    model.backbone.bert.pooler._trainable=False
    trainable = model.trainable_variables

    checkpoint_dir = './checkpoints/'+str(fold_nr)+"_fold"
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    @tf.function
    def train_step(inputs, y_true):
        with tf.GradientTape() as tape:
            ids_mask, type_ids_mask, attention_mask = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]
            y_pred = model(ids_mask, 
                             attention_mask= attention_mask, 
                             token_type_ids=type_ids_mask, 
                             training=True)
            loss = tf.reduce_sum(bce_loss(y_true, y_pred)*(1. / batch_size))

        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, trainable)

        return loss, grads, y_pred

    @tf.function
    def test_step(inputs, y_true):
        ids_mask, type_ids_mask, attention_mask = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]
        y_pred = model(ids_mask, 
                      attention_mask= attention_mask, 
                      token_type_ids=type_ids_mask, 
                      training=False)
        loss = tf.reduce_sum(bce_loss(y_true, y_pred)*(1. / batch_size))

        return loss, y_pred

    last_loss = 999999
    first_batch = True
    
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
            loss, current_gradient, y_pred = train_step(inputs, y_true)
            train_losses.append(np.mean(loss))
            train_preds.append(y_pred)
            train_targets.append(y_true)
            gradients = accumulated_gradients(gradients, current_gradient, gradient_accumulate_steps)

            if (batch_nr +1)%gradient_accumulate_steps ==0:
                #print("batch_nr {} gradient applying" .format(batch_nr))
                optimizer.apply_gradients(zip(gradients, trainable))
                global_batch +=1
                gradients = None

                if first_batch:
                    print("first batch")
                    hvd.broadcast_variables(trainable, root_rank=0)
                    hvd.broadcast_variables(optimizer.variables(), root_rank=0)
                    first_batch=False

            if batch_nr % 300 == 0 and hvd.local_rank() == 0:
                print('Step #%d\tLoss: %.6f' % (batch_nr, loss))

        for _, (inputs, y_true) in enumerate(test_ds):
            loss, y_pred = test_step(inputs, y_true)
            test_losses.append(np.mean(loss))
            test_preds.append(y_pred)
            test_targets.append(y_true)

        test_spearmans = spearman_metric(np.vstack(test_targets), np.vstack(test_preds))
        train_spearmans = spearman_metric(np.vstack(train_targets), np.vstack(train_preds))
        if hvd.local_rank() == 0:
            print("epoch {} train loss {} test loss {} test spearman {} train spearman {}" .format(epoch, np.mean(train_losses), np.mean(test_losses),  np.mean(test_spearmans), np.mean(train_spearmans)))
        if np.mean(test_spearmans) < last_loss:
            if hvd.rank() == 0:
                checkpoint.save(os.path.join(checkpoint_dir, "model_best"))
                last_loss = np.mean(test_spearmans)
                print("saving checkpoint... ")
    fold_nr += 1
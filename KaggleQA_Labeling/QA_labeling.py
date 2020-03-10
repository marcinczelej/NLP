import os
import numpy as np

import pandas as pd
import tensorflow as tf

from transformers import BertTokenizer, RobertaTokenizer
from BertModel import BertForQALabeling
from RoBERTaModel import *
from Trainer import Trainer
from preprocessing import dataPreprocessor
from parameters import *

import horovod.tensorflow as hvd

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
stack_df = pd.read_csv(os.path.join(data_dir, "stackexchange.csv"))
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

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

q_title = train_X_df['question_title'].values
q_body = train_X_df['question_body'].values
answer = train_X_df['answer'].values

stack_q_title = stack_df['question_title'].values
stack_q_body = stack_df['question_body'].values
stack_answer = stack_df['answer'].values

targets = train_targets_df.to_numpy()

dataPreprocessor.logger = False
dataPreprocessor.tokenizer = tokenizer
dataPreprocessor.model = "Roberta"

preprocessedInput = dataPreprocessor.preprocessBatch(q_body, q_title, answer, max_seq_lengths=(26,260,210,500))
preprocessedStack = dataPreprocessor.preprocessBatch(stack_q_body, stack_q_title, stack_answer, max_seq_lengths=(26,260,210,500))

Trainer.train(model=RoBERTaForQALabelingMultipleHeads.from_pretrained('roberta-base', num_labels=num_labels, output_hidden_states=True), 
              tokenizer=tokenizer, 
              preprocessedInput=preprocessedInput, 
              targets=targets, 
              preprocessedPseudo=preprocessedStack)
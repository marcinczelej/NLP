import os
import re
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def unicode_to_ascii(s):
  return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def normalize(s):
  s = unicode_to_ascii(s)
  s = re.sub(r'([!.?])', r' \1', s)
  s = re.sub(r'[^a-zA-Z.!?-]+', r' ', s)
  s = re.sub(r'\s+', r' ', s)
  return s

def read_data(data_dir, file_name):
  full_path = os.path.join(data_dir, file_name)
  print("reading data from ", full_path)

  with open(full_path) as file:
    lines = file.readlines()
  
  data = []

  for line in lines:
      data.append(line.split("\t")[:-1])
  
  return data

def read_data_files(data_dir, file_names):
    
  en_file_name, fr_file_name = file_names
  
  full_path = os.path.join(data_dir, en_file_name)
  print("reading data from ", full_path)

  with open(full_path) as file:
    en_lines = file.readlines()
  
  full_path = os.path.join(data_dir, fr_file_name)
  print("reading data from ", full_path)

  with open(full_path) as file:
    fr_lines = file.readlines()    
  
  return en_lines, fr_lines


def makeDatasets(train_data, test_data, batch_size, strategy=None):
  """
    Method that created distributed/not distributed train and test datasets that can be later feed
    into model in training/testing loop

    Parameters:
        train_data - input data for training. Should be in form : en_train, fr_train_in, fr_train_out
        test_data - input data for test step. Should be in form : en_test, fr_test_in, fr_test_out
        batch_size - batch_size that should be used to create datasets
        strategy - strategy that datasets should use to be distributed across GPUs. Default is None
    
    Returns:
      train_dataset - training dataset
      test_dataset - testing dataset
  """
  
  print("creating dataset...")
  en_train, fr_train_in, fr_train_out = train_data
  en_test, fr_test_in, fr_test_out = test_data
  
  train_dataset = tf.data.Dataset.from_tensor_slices((en_train, fr_train_in, fr_train_out))
  train_dataset = train_dataset.shuffle(len(en_train), reshuffle_each_iteration=True)\
                                    .batch(batch_size, drop_remainder=True)

  test_dataset = tf.data.Dataset.from_tensor_slices((en_test, fr_test_in, fr_test_out))
  test_dataset = test_dataset.shuffle(len(en_test), reshuffle_each_iteration=True)\
                                  .batch(batch_size, drop_remainder=True)
  
  if strategy is not None:
      train_dataset = strategy.experimental_distribute_dataset(train_dataset)
      test_dataset = strategy.experimental_distribute_dataset(test_dataset)
  
  return train_dataset, test_dataset
    
def loss_accuracy_plot(losses, accuracy, name = "", save_to_file=True):
  """
    Method to plot and save accuracy and loss plots using passed vectors

    Parameters:
      losses - tuple of losses containing (train_losses, test_losses)
      accuracy - tuple of accuracy containing (train_accuracy, test_accuracy)
      name - name that should appear in saved plot name
      save_to_file - should plots be saved to file. Default is set to True
  """

  if not os.path.exists("plots"):
    os.mkdir("plots")
  
  train_losses, test_losses = losses 
  train_accuracyVec, test_accuracyVec = accuracy

  fig = plt.figure()
  fig_plot = fig.add_subplot()
  fig_plot.plot(train_losses, label="train_loss")
  fig_plot.plot(test_losses, label="test_loss")
  fig_plot.legend(loc="upper right")
  fig_plot.set_xlabel("epoch")
  fig_plot.set_ylabel("loss")
  fig_plot.grid(linestyle="--")
  fig.show()

  acc_fig = plt.figure()
  acc_fig_plot = acc_fig.add_subplot()
  acc_fig_plot.plot(train_accuracyVec, label="train_accuracy")
  acc_fig_plot.plot(test_accuracyVec, label="test_accuracy")
  acc_fig_plot.legend(loc="lower right")
  acc_fig_plot.set_xlabel("epoch")
  acc_fig_plot.set_ylabel("accuracy")
  acc_fig_plot.grid(linestyle="--")
  acc_fig.show()

  if save_to_file:
    fig.savefig("plots/" + name + "_losses_plot.png")
    acc_fig.savefig("plots/" + name + "_accuracy_plot.png")
        
def save_to_csv(losses, accuracy, append, file_name):
  """
  Method to save accuracy and loss to csv file

  Parameters:
    losses - tuple of losses containing (train_losses, test_losses)
    accuracy - tuple of accuracy containing (train_accuracy, test_accuracy)
    append - should losses be appended to existing file
    file_name - where to save ata, file name
  """

  train_losses, test_losses = losses 
  train_accuracyVec, test_accuracyVec = accuracy

  if append and os.path.isfile(file_name):
      print("oppening ", file_name)
      df = pd.read_csv(file_name)
  else:
      print("creating new file: ", file_name)
      df = pd.DataFrame(columns=['train_loss', 'test_loss', 'train_acc', 'test_acc'])
  
  d = {'train_loss':[loss.numpy() for loss in train_losses], 
        'test_loss' :[loss.numpy() for loss in test_losses], 
        'train_acc' :[acc.numpy() for acc in train_accuracyVec], 
        'test_acc'  :[acc.numpy() for acc in test_accuracyVec]}
  df2 = pd.DataFrame(data = d)
  df = pd.concat([df, df2], ignore_index=True)
  df.to_csv("data.csv", index=False)

        
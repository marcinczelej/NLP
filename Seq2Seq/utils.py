import os
import re
import unicodedata

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def normalize(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s

# data can be downloaded from http://www.manythings.org/anki/
def read_data(data_dir, file_name):
    full_path = os.path.join(data_dir, file_name)

    with open(full_path) as file:
      lines = file.readlines()
    
    data = []
    
    for line in lines:
        data.append(line.split("\t")[:-1])
    
    return data


def preprocess(sequences, tokenizer):
  sequences = tokenizer.texts_to_sequences(sequences)

  max_len = None
  for seq in sequences:
    if max_len == None or len(seq) > max_len:
      max_len = len(seq)
  
  return pad_sequences(sequences, maxlen=max_len, padding='post')

def preprocessData(en_seq, fr_seq_in, fr_seq_out, fr_test, en_test):
  
  fr_tokenizer = Tokenizer(filters='')
  fr_tokenizer.fit_on_texts(fr_seq_in)
  fr_tokenizer.fit_on_texts(fr_seq_out)
  fr_tokenizer.fit_on_texts(fr_test)

  en_tokenizer = Tokenizer(filters='')
  en_tokenizer.fit_on_texts(en_seq)
  en_tokenizer.fit_on_texts(en_test)

  preprocessed_en_seq = preprocess(en_seq, en_tokenizer)

  preprocessed_fr_seq_in = preprocess(fr_seq_in, fr_tokenizer)
  preprocessed_fr_seq_out = preprocess(fr_seq_out, fr_tokenizer)

  return preprocessed_en_seq, preprocessed_fr_seq_in, preprocessed_fr_seq_out, en_tokenizer, fr_tokenizer 
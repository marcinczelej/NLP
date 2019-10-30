import unicodedata
import os
import re
import tensorflow

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

    with open(full_path) as file:
      lines = file.readlines()
    
    data = []
    
    for line in lines:
        data.append(line.split("\t")[:-1])
    
    return data

def preprocessSeq(texts, tokenizer):
  texts = tokenizer.texts_to_sequences(texts)

  return pad_sequences(texts, padding='post')

def tokenizeData(fr_train_in, fr_train_out, fr_test, en_train, en_test):
  fr_tokenizer = Tokenizer(filters='')

  fr_tokenizer.fit_on_texts(fr_train_in)
  fr_tokenizer.fit_on_texts(fr_train_out)
  fr_tokenizer.fit_on_texts(fr_test)

  fr_train_in = preprocessSeq(fr_train_in, fr_tokenizer)
  fr_train_out = preprocessSeq(fr_train_out, fr_tokenizer)
  fr_test_tokenized = preprocessSeq(fr_test, fr_tokenizer)

  en_tokenizer = Tokenizer(filters='')

  en_tokenizer.fit_on_texts(en_train)
  en_tokenizer.fit_on_texts(en_test)

  en_train = preprocessSeq(en_train, en_tokenizer)
  en_test = preprocessSeq(en_test, en_tokenizer)

  return (fr_tokenizer, en_tokenizer), (fr_train_in, fr_train_out, fr_test_tokenized), (en_train, en_test)
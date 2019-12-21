import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, r"../utilities")

from utils import *
from model import Encoder, Decoder
from Seq2SeqTrainer import Seq2SeqTrainer

LSTM_SIZE = 512
EMBEDDING_SIZE = 250
BATCH_SIZE= 64
EPOCHS = 600

def main():
    data_dir = "../data"
    # reading data

    data = read_data(os.path.join(data_dir, "fra-eng"), "fra.txt")
    en_lines, fr_lines = list(zip(*data))
    en_lines, fr_lines = shuffle(en_lines, fr_lines)

    en_lines = en_lines[:30000]
    fr_lines = fr_lines[:30000]

    en_lines = [normalize(line) for line in en_lines]
    fr_lines = [normalize(line) for line in fr_lines]

    en_train, en_test, fr_train, fr_test = train_test_split(en_lines, fr_lines, shuffle=True, test_size=0.1)

    en_lines = en_test
    fr_lines = fr_test

    fr_train_in = ['<start> ' + line for line in fr_train]
    fr_train_out = [line + ' <end>' for line in fr_train]

    fr_test_in = ['<start> ' + line for line in fr_test]
    fr_test_out = [line + ' <end>' for line in fr_test]

    fr_tokenizer = Tokenizer(filters='')
    en_tokenizer = Tokenizer(filters='')

    input_data = [fr_train_in, fr_train_out, fr_test_in, fr_test_out, fr_test, fr_train]
    fr_train_in, fr_train_out, fr_test_in, fr_test_out, fr_test, fr_train = tokenizeInput(input_data,
                                                                                          fr_tokenizer)
    input_data = [en_train, en_test]
    en_train, en_test = tokenizeInput(input_data, en_tokenizer)

    en_vocab_size = len(en_tokenizer.word_index)+1
    fr_vocab_size = len(fr_tokenizer.word_index)+1
    print("en_vocab {}\nfr_vocab {}" .format(en_vocab_size, fr_vocab_size))
    
    trainer = Seq2SeqTrainer(BATCH_SIZE, LSTM_SIZE, EMBEDDING_SIZE, predict_every=1)
    losses, accuracy = trainer.train([en_train, fr_train_in, fr_train_out], [en_test, fr_test_in, fr_test_out], [en_lines, fr_lines], [en_tokenizer, fr_tokenizer], 20)
    
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
    fig.savefig("losses_plot.png")
    fig.show()

    fig = plt.figure()
    fig_plot = fig.add_subplot()
    fig_plot.plot(train_accuracyVec, label="train_accuracy")
    fig_plot.plot(test_accuracyVec, label="test_accuracy")
    fig_plot.legend(loc="lower right")
    fig_plot.set_xlabel("epoch")
    fig_plot.set_ylabel("accuracy")
    fig_plot.grid(linestyle="--")
    fig.savefig("accuracy_plot.png")

if __name__ == "__main__":
    main()
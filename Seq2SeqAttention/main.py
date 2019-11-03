import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import *
from params import *
from Trainer import *

def main(args):
    LSTM_SIZE = args.lstm_units
    EMBEDDING_SIZE = args.embedding_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    files = args.file
    files = files.split(' ')
    print(files)
    if len(files) == 1:
        data = read_data(args.dir, args.file)
        en_lines, fr_lines = list(zip(*data))
        #en_lines=en_lines[:30000]
        #fr_lines=fr_lines[:30000]
    elif len(files) == 2:
        en_lines, fr_lines = read_data_files(args.dir, (files[0], files[1]))
    else:
        raise Exception("wrong number of files were given")
    print("Data reading: Done")
    
    en_lines = [normalize(line) for line in en_lines]
    fr_lines = [normalize(line) for line in fr_lines]

    en_train, en_test, fr_train, fr_test = train_test_split(en_lines, fr_lines, shuffle=True, test_size=0.05)

    fr_train_in = ['<start> ' + line for line in fr_train]
    fr_train_out = [line + ' <end>' for line in fr_train]
    
    tokenizers, fr_data, en_data = tokenizeData(fr_train_in, fr_train_out, fr_test, en_train, en_test)

    fr_tokenizer, en_tokenizer = tokenizers

    en_vocab_size = len(en_tokenizer.word_index)+1
    fr_vocab_size = len(fr_tokenizer.word_index)+1
    print("en_vocab_size {}\nfr_vocab_size {}" .format(en_vocab_size, fr_vocab_size))

    if args.training_type==1:
        print("Strating normal training loop with:\n  epochs {}\n  batch_szie {}\n  lstm_units {}\n  embedding_size {}\n  attention score {}"\
                .format(EPOCHS, BATCH_SIZE, LSTM_SIZE, EMBEDDING_SIZE, args.attention_type))
        train_losses, test_losses = train( \
            en_data, fr_data, en_test, fr_test, (en_vocab_size, fr_vocab_size), fr_tokenizer, en_tokenizer, args.attention_type)
    elif args.training_type==2:
        print("Starting distributed training loop with:\n  epochs {}\n  batch_szie {}\n  lstm_units {}\n  embedding_size {}\n  attention score {}"\
                .format(EPOCHS, BATCH_SIZE, LSTM_SIZE, EMBEDDING_SIZE, args.attention_type))
        train_losses, test_lossses = distributedTrain( \
            en_data, fr_data, en_test, fr_test, (en_vocab_size, fr_vocab_size), fr_tokenizer, en_tokenizer, args.attention_type)

    fig = plt.figure()
    fig_plot = fig.add_subplot()
    fig_plot.plot(train_losses, label="train_loss")
    fig_plot.plot(test_losses, label="test_loss")
    fig_plot.legend(location="lower left")
    fig_plot.set_xlabel("epoch")
    fig_plot.set_ylabel("loss")
    fig_plot.grid(linestyle="--")
    fig.savefig("losses_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="set value of BATCH_SIZE. Default=32")
    parser.add_argument('--embedding_size', type=int, default=256, help="set value of EMBEDDING_SIZE. Default=256")
    parser.add_argument('--lstm_units', type=int, default=512, help="set value of LSTM_UNITS. Default=512")
    parser.add_argument('--training_type',type=int,  default=1, help="set implementation to use : 1. single GPU/CPU training   2.distributed training    Default=1")
    parser.add_argument('--attention_type', type=str,  default="concat", help="set  attention score type to use: 1.dot   2.general   3.concat    Default=concat")
    parser.add_argument('--epochs', type=int, default=60, help="set EPOCHS number. Default=60")
    parser.add_argument('--dir', type=str, default="data/fra-eng", help="set directory with input data. Default=data/fra-eng")
    parser.add_argument('--file', type=str, default="fra.txt", help="set name of data files. For multiple files should be en_file_name fr_file_name Default=fra.txt")
    args = parser.parse_args()

    main(args)

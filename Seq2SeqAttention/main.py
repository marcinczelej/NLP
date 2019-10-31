import os
import argparse
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
    
    data = read_data(args.dir, args.file)
    print("Data reading: Done")
    en_lines, fr_lines = zip(*data)

    en_lines = [normalize(line) for line in en_lines]
    fr_lines = [normalize(line) for line in fr_lines]

    en_train, en_test, fr_train, fr_test = train_test_split(en_lines, fr_lines, shuffle=True, test_size=0.1)

    fr_train_in = ['<start> ' + line for line in fr_train]
    fr_train_out = [line + ' <end>' for line in fr_train]
    
    tokenizers, fr_data, en_data = tokenizeData(fr_train_in, fr_train_out, fr_test, en_train, en_test)

    fr_tokenizer, en_tokenizer = tokenizers

    en_vocab_size = len(en_tokenizer.word_index)+1
    fr_vocab_size = len(fr_tokenizer.word_index)+1
    print("en_vocab_size {}\nfr_vocab_size {}" .format(en_vocab_size, fr_vocab_size))
    
    print(type(args.training))
    if args.training==1:
        print("Strating normal training loop with:\n  epochs {}\n  batch_szie {}\n  lstm_units {}\n  embedding_size {}\n  attention score {}"\
                .format(EPOCHS, BATCH_SIZE, LSTM_SIZE, EMBEDDING_SIZE, args.implementation))
        train_losses, test_losses = train( \
            en_data, fr_data, fr_test, (en_vocab_size, fr_vocab_size), fr_tokenizer, args.implementation)
    elif args.training==2:
        print("Starting distributed training loop with:\n  epochs {}\n  batch_szie {}\n  lstm_units {}\n  embedding_size {}\n  attention score {}"\
                .format(EPOCHS, BATCH_SIZE, LSTM_SIZE, EMBEDDING_SIZE, args.implementation))
        train_losses, test_lossses = distributedTrain( \
            en_data, fr_data, fr_test, (en_vocab_size, fr_vocab_size), fr_tokenizer, args.implementation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="set value of BATCH_SIZE. Default=32")
    parser.add_argument('-em', '--embedding_size', type=int, default=256, help="set value of EMBEDDING_SIZE. Default=256")
    parser.add_argument('-u', '--lstm_units', type=int, default=512, help="set value of LSTM_UNITS. Default=512")
    parser.add_argument('-t', '--training',type=int,  default=1, help="set implementation to use : 1. single GPU/CPU training   2.distributed training    Default=1")
    parser.add_argument('-i', '--implementation', type=str,  default="concat", help="set  attention score type to use: 1.dot   2.general   3.concat    Default=concat")
    parser.add_argument('-e', '--epochs', type=int, default=60, help="set EPOCHS number. Default=60")
    parser.add_argument('-d', '--dir', type=str, default="data/fra-eng", help="set directory with input data. Default=data/fra-eng")
    parser.add_argument('-f', '--file', type=str, default="fra.txt", help="set name of data file. Default=fra.txt")
    args = parser.parse_args()

    main(args)

import os
import sys
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.insert(0, r"../utilities")

from utils import *
import params
from Trainer import *
from Seq2SeqAttentionTrainer import Seq2SeqAttentionTrainer

def main(args):
    params.LSTM_SIZE = args.lstm_units
    params.EMBEDDING_SIZE = args.embedding_size
    params.BATCH_SIZE = args.batch_size
    params.EPOCHS = args.epochs
    
    losses = [], []
    accuracy = [], []
   
    files = args.file
    files = files.split(' ')
    print(files)
    if len(files) == 1:
        data = read_data(args.dir, args.file)
        en_lines, fr_lines = list(zip(*data))
    elif len(files) == 2:
        en_lines, fr_lines = read_data_files(args.dir, (files[0], files[1]))
    else:
        raise Exception("wrong number of files were given")
    data_dir = "../data"
    en_lines, fr_lines = read_data_files(data_dir, ("small_vocab_en", "small_vocab_fr"))
    print("Data reading: Done")
    
    en_lines = [normalize(line) for line in en_lines]
    fr_lines = [normalize(line) for line in fr_lines]

    en_train, en_test, fr_train, fr_test = train_test_split(en_lines, fr_lines, shuffle=True, test_size=0.1)

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

    if args.training_type==1:
        print("Strating normal training loop with:\n  epochs {}\n  batch_szie {}\n  lstm_units {}\n  embedding_size {}\n  attention score {}"\
                .format(params.EPOCHS, params.BATCH_SIZE, params.LSTM_SIZE, params.EMBEDDING_SIZE, args.attention_type))
        #train_losses, test_losses = train( \
        #    en_data, fr_data, en_test, fr_test, fr_tokenizer, en_tokenizer, args.attention_type)
    elif args.training_type==2:
        trainer = Seq2SeqAttentionTrainer(params.BATCH_SIZE, params.LSTM_SIZE, params.EMBEDDING_SIZE, 1)
        print("Starting distributed training loop with:\n  epochs {}\n  batch_szie {}\n  lstm_units {}\n  embedding_size {}\n  attention score {}"\
                .format(params.EPOCHS, params.BATCH_SIZE, params.LSTM_SIZE, params.EMBEDDING_SIZE, args.attention_type))
        losses, accuracy = trainer.train([en_train, fr_train_in, fr_train_out], [en_test, fr_test_in, fr_test_out], [en_tokenizer, fr_tokenizer], params.EPOCHS, args.attention_type)
     
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="set value of BATCH_SIZE. Default=32")
    parser.add_argument('--embedding_size', type=int, default=256, help="set value of EMBEDDING_SIZE. Default=256")
    parser.add_argument('--lstm_units', type=int, default=512, help="set value of LSTM_UNITS. Default=512")
    parser.add_argument('--training_type',type=int,  default=2, help="set implementation to use : 1. single GPU/CPU training   2.distributed training    Default=2")
    parser.add_argument('--attention_type', type=str,  default="concat", help="set  attention score type to use: 1.dot   2.general   3.concat    Default=concat")
    parser.add_argument('--epochs', type=int, default=60, help="set EPOCHS number. Default=60")
    parser.add_argument('--dir', type=str, default="../data/fra-eng", help="set directory with input data. Default=data/fra-eng")
    parser.add_argument('--file', type=str, default="fra.txt", help="set name of data files. For multiple files should be en_file_name fr_file_name Default=fra.txt")
    args = parser.parse_args()

    main(args)

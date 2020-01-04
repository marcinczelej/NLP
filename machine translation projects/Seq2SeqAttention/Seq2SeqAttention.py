import os
import sys
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, r"../utilities")

from utils import *
from Seq2SeqAttentionTrainer import Seq2SeqAttentionTrainer

tf.get_logger().setLevel('WARNING')

def main(args):
    
    LSTM_SIZE = args.lstm_units
    EMBEDDING_SIZE = args.embedding_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    data_dir = "../data"
    en_lines, fr_lines = read_data_files(data_dir, ("small_vocab_en", "small_vocab_fr"))

    #data = read_data(os.path.join(data_dir, "fra-eng"), "fra.txt")

    #en_lines, fr_lines = list(zip(*data))
    #en_lines, fr_lines = shuffle(en_lines, fr_lines)

    #en_lines = en_lines[:30000]
    #fr_lines = fr_lines[:30000]

    en_lines = [normalize(line) for line in en_lines]
    fr_lines = [normalize(line) for line in fr_lines]

    en_train, en_test, fr_train, fr_test = train_test_split(en_lines, fr_lines, shuffle=True, test_size=0.1)

    en_lines = en_test
    fr_lines = fr_test

    # creating tokenizers
    en_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en for en in en_train), target_vocab_size=2**13)

    fr_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (fr for fr in fr_train), target_vocab_size=2**13)

    print("en_tokenizer size ", en_tokenizer.vocab_size)
    print("fr_tokenizer size ", fr_tokenizer.vocab_size)

    en_tokenizer.save_to_file("en_tokenizer")
    fr_tokenizer.save_to_file("fr_tokenizer")

    # train dataset
    fr_train_in = [[fr_tokenizer.vocab_size] + fr_tokenizer.encode(line) for line in fr_train]
    fr_train_out = [fr_tokenizer.encode(line) + [fr_tokenizer.vocab_size+1] for line in fr_train]

    fr_train_in = pad_sequences(fr_train_in, padding='post')
    fr_train_out = pad_sequences(fr_train_out, padding='post')

    # test dataset
    fr_test_in = [[fr_tokenizer.vocab_size] + fr_tokenizer.encode(line) for line in fr_test]
    fr_test_out = [fr_tokenizer.encode(line) + [fr_tokenizer.vocab_size+1] for line in fr_test]

    fr_test_in = pad_sequences(fr_test_in, padding='post')
    fr_test_out = pad_sequences(fr_test_out, padding='post')

    en_train = [en_tokenizer.encode(line) for line in en_train]
    en_test = [en_tokenizer.encode(line) for line in en_test]

    en_train = pad_sequences(en_train, padding='post')
    en_test = pad_sequences(en_test, padding='post')

    trainer = Seq2SeqAttentionTrainer(batch_size=BATCH_SIZE, 
                                      lstm_size=LSTM_SIZE, 
                                      embedding_size=EMBEDDING_SIZE, 
                                      tokenizers=[en_tokenizer, fr_tokenizer],
                                      predict_every=5)

    losses, accuracy= trainer.train(train_data=[en_train, fr_train_in, fr_train_out], 
                                    test_data=[en_test, fr_test_in, fr_test_out], 
                                    prediction_data=[en_lines, fr_lines], 
                                    epochs=EPOCHS, 
                                    attention_type="concat")
     
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
    parser.add_argument('--attention_type', type=str,  default="concat", help="set  attention score type to use: 1.dot   2.general   3.concat    Default=concat")
    parser.add_argument('--epochs', type=int, default=20, help="set EPOCHS number. Default=20")
    args = parser.parse_args()

    main(args)

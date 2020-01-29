import argparse
import sys

import tensorflow as tf

sys.path.insert(0, r"../utilities")
sys.path.insert(0, r"../preprocessing")
sys.path.insert(0, r"../trainers")

from preprocessor import *
from utils import loss_accuracy_plot

from trainers.Seq2SeqAttentionTrainer import Seq2SeqAttentionTrainer

tf.get_logger().setLevel('WARNING')

def main(args):
    
    LSTM_SIZE = args.lstm_units
    EMBEDDING_SIZE = args.embedding_size
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    
    encoder_tokenizer_name = args.encoder_tokenizer
    decoder_tokenizer_name = args.decoder_tokenizer
    
    data_dir = "../data"
    en_lines, fr_lines = read_data_files(data_dir, ("small_vocab_en", "small_vocab_fr"))

    train_data, test_data, prediction_data, tokenizers = preprocess_data(en_lines, fr_lines, [encoder_tokenizer_name, decoder_tokenizer_name])

    trainer = Seq2SeqAttentionTrainer(batch_size=BATCH_SIZE, 
                                      lstm_size=LSTM_SIZE, 
                                      embedding_size=EMBEDDING_SIZE, 
                                      tokenizers=tokenizers,
                                      predict_every=5)

    losses, accuracy= trainer.train(train_data=train_data,
                                    test_data=test_data,
                                    prediction_data=prediction_data,
                                    epochs=EPOCHS, 
                                    attention_type="concat")

    loss_accuracy_plot(losses, accuracy, "Seq2SeqAttention")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help="set value of BATCH_SIZE. Default=32")
    parser.add_argument('--embedding_size', type=int, default=256, help="set value of EMBEDDING_SIZE. Default=256")
    parser.add_argument('--lstm_units', type=int, default=512, help="set value of LSTM_UNITS. Default=512")
    parser.add_argument('--attention_type', type=str,  default="concat", help="set  attention score type to use: 1.dot   2.general   3.concat    Default=concat")
    parser.add_argument('--epochs', type=int, default=20, help="set EPOCHS number. Default=20")
    parser.add_argument('--encoder_tokenizer', type=str, default="en_tokenizer", help="sets name of encoder_tokenizer to load/save in case when no such tokenizer exists. Default=en_tokenizer")
    parser.add_argument('--decoder_tokenizer', type=str, default="fr_tokenizer", help="sets name of decoder tokenizer to load/save in case when no such tokenizer exists. Default=fr_tokenizer")
    args = parser.parse_args()

    main(args)

import copy

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tokenizer import get_tokenizers
from utils import *

def preprocess_data(inputs, outputs, tokenizer_names, shuffle_needed=True):
    """
        Method for input and output data preprocessing.
        It consist of following steps:
        1. shuffling, if shuffle set to True. Default value : True
        1. normalizing data
        2. splitting data into train/test
        3. creating/loading tokenizers
        4. adding <start> and <end> tokens into train/test data
        5. padding train/test data
        
        Parameters:
            inputs - data that needs to be translated
            outputs - target translations
            tokenizer_names - names of tokenizer files
            shuffle_needed - should data be shuffled. Default value : True
        
        Returns:
            train_data - data vector consting preprocessed train data in form: (input, output_in, output_out)
            test_data - data vector consting preprocessed test data in form: (input, output_in, output_out)
            prediction_data - only normalized intput, output datas
            tokenizers - tokenizers for input and output languages
    """
    
    input_tokenizer_name, output_tokenizer_name = tokenizer_names
    
    input_data = copy.deepcopy(inputs)
    output_data = copy.deepcopy(outputs)
    
    if shuffle_needed:
        input_data, output_data = shuffle(input_data, output_data)
    
    input_data = [normalize(line) for line in input_data]
    output_data = [normalize(line) for line in output_data]

    input_train, intput_test, output_train, output_test = train_test_split(input_data, output_data, shuffle=True, test_size=0.1)

    intput_lines = intput_test
    output_lines = output_test
    
    tokenizer_data = {input_tokenizer_name : input_train,
                 output_tokenizer_name : output_train}
    input_tokenizer, output_tokenizer = get_tokenizers(tokenizer_data)
    print("Tokenizers created\n  {} vocab size {}\n  {} vocab size {}" \
      .format(input_tokenizer_name, input_tokenizer.vocab_size, \
              output_tokenizer_name, output_tokenizer.vocab_size))
    
    # train dataset
    output_train_in = [[output_tokenizer.vocab_size] + output_tokenizer.encode(line) for line in output_train]
    output_train_out = [output_tokenizer.encode(line) + [output_tokenizer.vocab_size+1] for line in output_train]

    output_train_in = pad_sequences(output_train_in, padding='post')
    output_train_out = pad_sequences(output_train_out, padding='post')

    # test dataset
    output_test_in = [[output_tokenizer.vocab_size] + output_tokenizer.encode(line) for line in output_test]
    output_test_out = [output_tokenizer.encode(line) + [output_tokenizer.vocab_size+1] for line in output_test]

    output_test_in = pad_sequences(output_test_in, padding='post')
    output_test_out = pad_sequences(output_test_out, padding='post')

    input_train = [input_tokenizer.encode(line) for line in input_train]
    intput_test = [input_tokenizer.encode(line) for line in intput_test]

    input_train = pad_sequences(input_train, padding='post')
    intput_test = pad_sequences(intput_test, padding='post')

    train_data = [input_train, output_train_in, output_train_out]
    test_data = [intput_test, output_test_in, output_test_out]
    prediction_data = [intput_lines, output_lines]
    tokenizers = [input_tokenizer, output_tokenizer]

    return train_data, test_data, prediction_data, tokenizers
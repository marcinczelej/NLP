import os
import pathlib
import tensorflow_datasets as tfds

def get_tokenizers(input_data):
    """
        Method that returns proper tokenizers for given input data
        If there is proper tokenizer file, given as dictionary key,
        this method will load tokenizer from it. If not it will
        create new tokenizer and save it to proper file in this directory
        
        Parameters:
            input_data - input datas that are used to build tokenizers. 
                         Should be given in form of dictionary:
                         tokenizer_file_name : data
        Returns:
            tokenizers - vector of tokenizers
    """

    # creating tokenizers
    tokenizers = []
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for file_name, data in input_data.items():
        file_name = current_dir + "/" + file_name
        if pathlib.Path(file_name + ".subwords").exists():
            print("loading tokenizer from file ", file_name + ".subwords")
            tokenizer=tfds.features.text.SubwordTextEncoder.load_from_file(file_name)
        else:
            print("creating new tokenizer")
            tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (text for text in data), target_vocab_size=2**13)
            tokenizer.save_to_file(file_name)
        tokenizers.append(tokenizer)

    return tokenizers
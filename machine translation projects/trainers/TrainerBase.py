from abc import abc, abstractmethod

class BaseTrainer(abc):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def train(self, train_data, test_data, prediction_data, epochs, restore_checkpoint, csv_name):
        pass

    @abstractmethod
    def translate(self, en_sentence):
        pass
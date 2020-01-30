from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
        Base class that should be used to create new Trainers
        Two methods are needed to implement in order to use it :
        - train
        - translate
    """

    @abstractmethod
    def train(self, train_data, test_data, prediction_data, epochs, restore_checkpoint, csv_name):
        pass

    @abstractmethod
    def translate(self, en_sentence):
        pass
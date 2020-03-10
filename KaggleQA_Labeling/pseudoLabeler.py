import pandas as pd
import tensorflow as tf
import os

class PseudoLabeler(object):
    """
        Class responsible for pseudo labeling data
        
        Parameters:
            checkpoint - checkopoint object used to load best saved model
            model - nn model that will be used for prediction
            optimizer - parameter needed for checkpoint 
            checkpoint_dir - directory of checkpoint to be loaded
            pseudo_labeling_df - dataframe with preprocessed input to be pseudolabeled. Preprocessing should be done with preprocessing.dataPreprocessor class
            fold_nr - fold number
        
        Pseudolabeling informations: 
            https://datawhatnow.com/pseudo-labeling-semi-supervised-learning/
    """
    @classmethod
    def create_pseudo_labels(cls, checkpoint, model, optimizer, checkpoint_dir, pseudo_labeling_df, fold_nr):
    
        checkpoint.restore(checkpoint_dir).assert_consumed()
        print("best checkpoint restored ...")
        
        pseudo_predictions = []
        for _, inputs in enumerate(pseudo_labeling_df):
            ids_mask, type_ids_mask, attention_mask = inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]
            predicted = model(ids_mask, 
                          attention_mask= attention_mask, 
                          token_type_ids=type_ids_mask, 
                          training=False)

            pseudo_predictions.extend(predicted.numpy())

        print("predicting pseudo=labels done ...")
        pseudo_predictions = tf.math.sigmoid(pseudo_predictions)
        predicted_df = stack_df.copy(deep=True)
        for idx, col in enumerate(train_targets_df.columns):
            predicted_df[col] = pseudo_predictions[:, idx]

        predicted_df.to_csv(
            os.path.join('./dataframes/pseudo_labeled' "best_base_uncased_fold-{}.csv" .format(fold_nr)), index=False)
        print("pseudo labeling for fold {} done " .format(fold_nr))
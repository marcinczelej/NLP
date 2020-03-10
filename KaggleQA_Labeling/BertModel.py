import tensorflow as tf
import numpy as np

from transformers.modeling_tf_bert import TFBertPreTrainedModel, TFBertModel
from transformers.modeling_tf_utils import get_initializer

class WeightedSumLayer(tf.keras.layers.Layer):
  def __init__(self, hidden_layers_num):
    super(WeightedSumLayer, self).__init__()

    self.hidden_states_weights = tf.Variable(initial_value=[-3.0]*hidden_layers_num + [0.0], dtype='float32', trainable=True, name="hidden_state_weights")
    self.softmax_act = tf.keras.layers.Softmax(axis=0)
  
  def call(self, inputs):
    output = tf.math.reduce_sum(self.softmax_act(self.hidden_states_weights)*inputs, axis=-1)
    return output

class BertForQALabeling(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        
        self.num_layers = config.num_labels
        self.backbone = TFBertModel(config,*inputs, **kwargs, name="bert_backbone")
        
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dropout_multisampled = tf.keras.layers.Dropout(0.5)
        
        self.weighted_sum = WeightedSumLayer(config.num_hidden_layers)
        
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier")
        
        self.backbone.bert.pooler._trainable=False
    
    def getName(self):
        return 'BertForQALabeling'
    
    def call(self, input_ids,
              attention_mask=None,
              token_type_ids=None, 
              **kwargs):
        
        bert_output = self.backbone(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids, 
                                    **kwargs)
        
        # https://huggingface.co/transformers/model_doc/bert.html#tfbertmodel
        # it`s better to use weighted average of all hiden states outputs
        # shape (batch_size, sequence_length, hidden_state)
        hidden_states = bert_output[2]
        
        transformed_hidden_states = tf.stack([self.dropout(hidden_state[:, 0, :], training=kwargs.get("training", False)) for hidden_state in hidden_states], axis = 2)
        transformed_hidden_states = self.weighted_sum(transformed_hidden_states)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788

        multisampled_logits = [self.classifier(self.dropout_multisampled(transformed_hidden_states, training=kwargs.get("training", False))) for _ in range(5)]
        stacked_logits = tf.stack(multisampled_logits, axis=0)

        output_logits = tf.math.reduce_mean(stacked_logits, axis=0)
        
        return output_logits
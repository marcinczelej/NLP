from math import ceil, floor

import tensorflow as tf

class dataPreprocessor(object):
    """
        Class for preprocessing input data so it will be in BERT format
    """
    logger = False
    tokenizer = None
    model = None
    
    @classmethod
    def combine_data(cls, q_body, q_title, answer, max_seq_length):
        """
            Method for combining preprocessed data with tokens into form:
            
            [CLS]question_title[SEP]question_body[SEP]answer[SEP]
            
            Additionally it will pad data to max_seq_length
            
            Robert will add additional tokenso it will add 4 tokens, when Bert will add only 3 tokens

            Returns:
              ids of input
        """

        question_data = q_title +[cls.tokenizer.sep_token_id]+ q_body
        answer_data = answer
        
        tokenizer_ids = cls.tokenizer.build_inputs_with_special_tokens(question_data, answer_data)
        tokenizer_type_ids = cls.tokenizer.create_token_type_ids_from_sequences(question_data, answer_data)
        attention_mask = [1] * len(tokenizer_ids)
        
        # padding
        tokenizer_ids = tokenizer_ids + [0]*(max_seq_length - len(tokenizer_ids))
        tokenizer_type_ids = tokenizer_type_ids + [0]*(max_seq_length - len(tokenizer_type_ids))
        attention_mask = attention_mask + [0]*(max_seq_length - len(attention_mask))

        return tokenizer_ids, tokenizer_type_ids, attention_mask
    
    @classmethod
    def cut_data_lengths(cls, q_body, q_title, answer, max_seq_lengths):
        """
            Method that should cut each part of input data to fit max_seq_length 
            1.Encode data and convert from binary type to utf-8
            2. do preprocessing and head-tial adding
            3. combine data with [CLS] and [SEP]s
            4. return preprred data
        """
        
        q_title = cls.tokenizer.encode(q_title, add_special_tokens=False)
        q_body = cls.tokenizer.encode(q_body, add_special_tokens=False)
        answer = cls.tokenizer.encode(answer, add_special_tokens=False)
        """
        if cls.logger:
            print("sep token ", cls.tokenizer.sep_token_id)
            print("pad token ", cls.tokenizer.pad_token_id)
            print("cls token ", cls.tokenizer.cls_token_id)
            print("q_title {}\n\nq_body{}\n\nanswer{}\n\n" .format(q_title, q_body, answer))
            print("q_title{}\nq_body {}\nanswer {}\n" .format(len(q_title), len(q_body), len(answer)))
        """
        
        max_q_title_len, max_q_body_len, max_answer_len, max_seq_length = max_seq_lengths
        
        # we`re substracting 4 due to tokens addition after preprocessing
        # if length is ok just add tokens
        if len(q_title) + len(q_body) + len(answer) <= max_seq_length -4:
            if cls.model == "Roberta":
                q_title = q_title[:-1]
            ids_mask, type_ids_mask, attention_mask = cls.combine_data(q_body=q_body, 
                                             q_title=q_title, 
                                             answer=answer,
                                             max_seq_length=max_seq_length)
            

            return (ids_mask, type_ids_mask, attention_mask)
            
        # if question title is shorter than max length:
        # add difference divided bu two to max length of both question body and answer body
        if len(q_title) < max_q_title_len:
            new_max_q_title_len = len(q_title)
            diff = max_q_title_len - len(q_title)
            max_q_body_len = max_q_body_len + ceil((diff)/2)
            max_answer_len = max_answer_len + floor((diff)/2)
        else:
            new_max_q_title_len = max_q_title_len
        
        # if question body length is shorter than max question body length:
        # add difference to max anser body length
        # else vice versa
        if len(q_body) < max_q_body_len:
            new_max_answer_len = max_answer_len + (max_q_body_len - len(q_body))
            new_max_q_body_len = len(q_body)
        elif len(answer) < max_answer_len:
            new_max_q_body_len = max_q_body_len + (max_answer_len - len(answer))
            new_max_answer_len = len(answer)
        else:
            new_max_answer_len = max_answer_len
            new_max_q_body_len = max_q_body_len
        
        # sanity check
        if (new_max_answer_len + new_max_q_body_len + new_max_q_title_len + 4) != max_seq_length:
            raise ValueError("Wrong sequence length  {} {} {} {} " \
                             .format(new_max_answer_len, new_max_q_body_len, new_max_q_title_len, max_seq_length))
        
        # paper : https://arxiv.org/pdf/1905.05583.pdf
        # head - Tail method
        # we`re taking input max number of elements as head
        # we`re taking remaining available elements as tail
        # we`re taking 1/2 as head 1/2 as tail
        head_q_body = round((new_max_q_body_len/2))
        tail_q_body = (new_max_q_body_len - head_q_body)
        
        head_answer = round((new_max_answer_len)/2)
        tail_answer = (new_max_answer_len - head_answer)
        
        head_q_title = max_q_title_len
        
        q_title = q_title[:head_q_title]
        q_body = q_body[:head_q_body] + q_body[-1*tail_q_body:]
        answer = answer[:head_answer] + answer[-1*tail_answer:]
        
        if cls.model == "Roberta":
                q_title = q_title[:-1]
        
        ids_mask, type_ids_mask, attention_mask = cls.combine_data(q_body=q_body, 
                                         q_title=q_title, 
                                         answer=answer, 
                                         max_seq_length=max_seq_length)
        
        return (ids_mask, type_ids_mask, attention_mask)
    
    @classmethod
    def preprocessBatch(cls, q_body, q_title, answer, max_seq_lengths):
        """
            Method for iterating through Batch in order to preprocess data
        """
        preprocessed_batch = None
        for i in range (q_body.shape[0]):
            preprocessed_element = tf.constant(cls.cut_data_lengths(q_body[i], q_title[i], answer[i], max_seq_lengths))[None,:]
            if preprocessed_batch == None:
                preprocessed_batch = preprocessed_element
            else:
                preprocessed_batch = tf.concat([preprocessed_batch, preprocessed_element], axis=0)
        return preprocessed_batch
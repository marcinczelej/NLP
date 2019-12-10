# Seq2Seq model

### TODO
- [ ] write report
- [ ] add results to report
- [ ] add beam search
- [ ] add bleu score
- [ ] make collab notebook

### Overview
Seq2Seq model with Attention  is encoder-decoder machine learning alghoritm that uses Attention mechanism and  can be used in
 - Machine Translation
 - Text Summarization
 - Conversational Modeling
 - Image Captioning
 - and in every task when we have sequenced input and want to get some output

Classic Seq2Seq models have problem with processing long sequence, because thwy base on single, fixed length context vector. Additionally those models look at model we want to translate only one time, as a whole, and use this input to produce every part of output. Attention models allow to look at every single part of input and based on that they produce decoder output.

Just line in Seq2Seq models without attention, Encoder part of Attention model, take input sentence we want to translate and during each timestep  it produces hidden states, one for ach sequence input. And here big difference appears.

In model without attention, only last hidden state vector is passed to Decoder, and becuase of that it has fixed length size (one hidden size), what leads to not so good results when handling longer sequences.
In model with attention Decoder gets all of the hidden states that encoder produces. Benefit of this is visible when we`re handling longer sequences. Becuase we don`t have fixed length context vector, so longer sentences will have longer context vectors that will capture input sequence information better than fixed length vector.

If we go one step further, we can say that becuase we have variable length of context vector, and each hidden state mostly captures information about corresponding elelment from input sequence. For example if we have 4 word sequence , then first hidden state captures first word the most, second second etc..
Additionally each hidden state, because it was generated using LSTM/GRU, captures informations about whole snetence, word dependencies etc. so we have many more informations than in vanillia Seq2Seq models.

#TODO: ADD GENERAL PICTURE
![Seq2Seq overview](https://smerity.com/media/images/articles/2016/gnmt_arch_1_enc_dec.svg)
[Source](https://smerity.com/articles/2016/google_nmt_arch.html)

We can see that there are two inputs:
- encoder Original sentence input
- decoder Desired translation input

And one output:
- decoder translated output

### Colab notebook

link to Colab notebook can be found [here](https://github.com/mizzmir/NLP/blob/master/machine%20translation%20projects/Seq2Seq/Seq2SeqColab.ipynb)

### Encoder

Encoder part is the same as in vanillia Seq2Seq model, with only one difference: output data.
Encoder part is straightforward and is build from Embedding layer  + LSTM layers that **returns all hidden states**. Those info are then passed to decoder as input, with decoder Desired translation input. As for encoder input, we get hidden states + input sequence we want to encode.

Let first forcus on structure of encoder, on what we need and what shapes it will have during each step of encoding process (generally for me, writing shapes can help to understand what`s going on inside and check if implementation is good or not)

![Seq2Seq Encoder shapes](../imgs/Seq2Seq/Encoder_shapes.jpg)

No now when we have information about shapes it is easy to implement:

```python
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_size, units):
    super(Encoder, self).__init__()

    self.units = units
    self.embeding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True, trainable=True)
    self.lstm_layer = tf.keras.layers.LSTM(units, dropout=0.2, return_sequences=True, return_state=True)
  
  def call(self, sequences, lstm_states):
    # sequences shape = [batch_size, seq_max_len]
    # lstm_states = [batch_size, lstm_size] x 2
    # encoder_embedded shape = [batch_size, seq_max_len, embedding_size]
    # output shape = [batch_size, seq_max_len, lstm_size]
    # state_h, state_c shape = [batch_size, lstm_size] x 2

    encoder_embedded = self.embeding_layer(sequences)
    #print("encoder_embedded = ", encoder_embedded.shape)
    output, state_h, state_c = self.lstm_layer(encoder_embedded, initial_state=lstm_states)

    return output, state_h, state_c

  def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.units]),
                tf.zeros([batch_size, self.units]))
```

Because we have to get initial hiddent states for encoder, that are all zeros at the start of every training step, there is method init_states(...) added. It returns properly shaped hidden states. In case of using GRU layer, there will be only one hidden state , instead of two.

#TODO: ADD PICTURES
### Attention mechanism

In vanilia Seq2Seq model encoder information is passed only to first node/ first timestep of Decoder network. due to that this information will become less and less relevant every next time step. To solve this we would like to have encoder output available at every time step of decoder and this is main idea behind attention.

We do this by creating **context vector**. Let me explain math behind it

1. We have to calculate so called **score** function

here we have few options to choose from, as can be seen below:

![Score functions](../imgs/Seq2SeqAttention/score_eq.png)

where:
ht - current decoder output state
hs - all source states given by encoder

Scoring function assigns score to each of the hidden states at given timestep "t". Higher the score, more important given hidden state is for given decoder timestep.

2. When we have our score of each encoder hidden states, we can calculate so called **alignment vector**

Alignment vector tells us what is propability/importance of each element using score value we calculated before.
It`s main purpose is to say how important/ assigns weights for each hidden state of encoder for given timestep "t" in decoder. The highest propability, the most important given element is.

It`s done by taking softmax of score function. After this we have vector, that is same length as source sequence, with propability/importance of each element. Shape of this vector should be intuitive, because we want some way to measure what hidden state to focus at given timestep, so we have to take weighted average ( what is done in next step) of all hidden states and to do this we need some kind of weights.

![alingment vector](../imgs/Seq2SeqAttention/alignment_eq.png)

3. When we have importance weights for each element, it`s time for calculating final step of our attention mechanism: **context vector**
'
Context vector is done by simply multiplying **alignment vector** with **encoder output**. This operation is nothing more than weighted average of encoder output. With this operation we have attention values for given timestep.

![Context vector](../imgs/Seq2SeqAttention/context_vector.png)

Because in attention we`re calculating output for each output timestep, it`s nice to visualise shapes again:

![Attention](../imgs/Seq2SeqAttention/Luang_attention_shapes.jpg)

Main difference is between score functions we choose to use, but later on calculations are straightforward.

* `dot` score function is best suited when both encoder and decoder have same embedding size/embedding space
* `general` score function is better suited when encoder and decoder have different embedding space, that's the case when we`re doing language translation. The reason for that is weights matrix that is added between multiplication of encoder and decoder states.
* `concat` can be use in all cases

Whole attention claculations are done with below class:

```python
class LuangAttention(tf.keras.Model):
def __init__(self, lstm_size, attention_type):
    super(LuangAttention, self).__init__()

    self.W_a = tf.keras.layers.Dense(lstm_size, name="LuangAttention_W_a")
    self.W_a_tanh = tf.keras.layers.Dense(lstm_size, activation="tanh", name="LuangAttention_W_a_tanh")
    self.v_a = tf.keras.layers.Dense(1)
    self.type = attention_type

def call(self, decoder_output, encoder_output):
    # encoder_output shape [batch_size, seq_max_len, hidden_units_of_encoder]
    # decoder_output shape [batch_size, 1, hidden_units of decoder]
    # score shape [batch_size, 1, seq_max_len]
    if self.type == "dot":
        score = tf.matmul(decoder_output, encoder_output, transpose_b=True)
    elif self.type == "general":
        score = tf.matmul(decoder_output, self.W_a(encoder_output), transpose_b=True)
    elif self.type == "concat":
        decoder_output = tf.broadcast_to(decoder_output, encoder_output.shape)
        concated = self.W_a_tanh(tf.concat((decoder_output, encoder_output), axis=-1))
        score = tf.transpose(self.v_a(concated), [0,2,1])
    else:
        raise Exception("wrong score function selected")
        
    alignment_vector = tf.nn.softmax(score, axis=2)
    context_vector = tf.matmul(alignment_vector, encoder_output)

    return context_vector, alignment_vector
```
#TODO: WRITE
### Decoder

Decoder part is where the Attention magic happens. In vanillia Seq2Seq decoder was build same way as encoder, but with differnet input. It also looks at whole sequence at once and based on this one look + contex vector from encoder (last hidden state) it creates translation output.

In Attention case there are few differences that I will adress here.

### Input Preprocessing

Preprocessing is  process that has to be done so we can push data into our model. It consists of few parts:

- **data normalization**

    In this step we're assuring that all sentences are in ascii format, cleaning wunwanted tokens, spaces before punctuations, changing to lowercase etc. Mostly constains general cleanup of text. It`s common to use two below methods (usually it's enough but somethins you want to add something extra for example leave some language specific characters or to leave some tokens)

    ```python
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')

    def normalize(s):
        s = unicode_to_ascii(s)
        s = re.sub(r'([!.?])', r' \1', s)
        s = re.sub(r'[^a-zA-Z.!?-]+', r' ', s)
        s = re.sub(r'\s+', r' ', s)
        return s
    ```

- **splitting data into train/test set and expanding desired decoder intput/output sequences**
It's done by adding `<start>` or `<end>` token respectively.

- **padding and tokenization**
    Entences are zero padded, so they will be same length, and tokenized into vectors of tokens (integers) with choosen Tokenizer.
    In this case build in tensorflow tokenizer wa used, but one can use nltk tokenizer, scipy tokenizer etc..
    We have to save both input and output language tokenizers to de-tokenize sentences later in prediction phase.

    ```python
    def preprocessSeq(texts, tokenizer):
    texts = tokenizer.texts_to_sequences(texts)

    return pad_sequences(texts, padding='post')

    def tokenizeInput(input_data, tokenizer):
        output_data = []
        for data in input_data:
            tokenizer.fit_on_texts(data)

        for data in input_data:
            output_data.append(preprocessSeq(data, tokenizer))

        return output_data
    ```

subroutines can be found [here](https://github.com/mizzmir/NLP/blob/master/machine%20translation%20projects/utilities/utils.py)

whole preprocess routine can be found [here](https://github.com/mizzmir/NLP/blob/master/machine%20translation%20projects/Seq2SeqAttention/main.py) plus code is pasted below:

```python
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
```

### training loop

Now lets talk about training loop. In order to make use of multiple gpus, few things has to be done. Custom training loop using multiple GPUs in tensorflow 2.0 is nicely described [here](https://www.tensorflow.org/tutorials/distribute/custom_training)

In order to use multiple GPU-s we have to create MirroredStrategy and then do whole training under its scope. Additionally we cannot use normal t.Datasets, becuase we want to "distribute it over multiple models on different GPUs. To do this we have to do two things:

1. set desired **BATCH_SIZE** for all models
2. use `strategy.experimental_distribute_dataset` 

As for first, we have to multiply desired BATCH_SIZE that we want to pass to single model, with number of GPUs we want to use. We can do this by simple `BATCH_SIZE * GUP_number` multiplication to use fixed number of GPUs, or use `strategy.num_replicas_in_sync` that will give us all available GPUs.

```python
    print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
    GLOBAL_BATCH_SIZE = self.batch_size*self.strategy.num_replicas_in_sync
```

Where:
    `self.strategy = tf.distribute.MirroredStrategy()`

After this , during ach training step GLOBAL_BACH_SIZE samples wil be taken from dataset and distributed among all models, so we each model will get BATCH_SIZE batches that we want.

Now when we have out desired GLOBAL_BATCH_SIZE let's create train/test datasets. Because we`re using distributed training our desired batch_size will be **GLOBAL_BATCH_SIZE**.

```python
        train_dataset = tf.data.Dataset.from_tensor_slices((en_train, fr_train_in, fr_train_out))
        train_dataset = train_dataset.shuffle(len(en_train), reshuffle_each_iteration=True)\
                                        .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        train_dataset_distr = self.strategy.experimental_distribute_dataset(train_dataset)

        test_dataset = tf.data.Dataset.from_tensor_slices((en_test, fr_test_in, fr_test_out))
        test_dataset = test_dataset.shuffle(len(en_test), reshuffle_each_iteration=True)\
                                       .batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        test_dataset_distr = self.strategy.experimental_distribute_dataset(test_dataset)
```

Only different thing , from non-distributed datasets are last lines for both test/train datasets

From this point everything we'll do, will be done under the score of strategy.MirroredStrategy().

1. **We have to create Encoder/Decoder/Optimizer under strategy scope:**

```python
            self.optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)
            self.encoder = Encoder(en_vocab_size, self.embedding_size, self.lstm_size)
            self.decoder = Decoder(fr_vocab_size, self.embedding_size, self.lstm_size)
```

2. **Loss function**

The next thing to do is to define a loss function. Because sequence is padded with zeros, we cannot take it into account when calculating loss. This will be handled with proper mask:

```python
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                                     reduction="none") 
            def compute_loss(predictions, labels):
                mask = tf.math.logical_not(tf.math.equal(labels, 0))
                mask = tf.cast(mask, tf.int64)
                per_example_loss = loss_obj(labels, predictions, sample_weight=mask)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
```

We're using `from_logit=True` because decoder output is not after softmax activation, so we're not passing propabilities, just values.
mask is used to zero padded values and is passed to loss obj with `sample_weight` parameter. Same result can be obtainet by multiplying predicted data by mask and then passing result to `loss_obj`

Because we`re using ditributed training we have to take average loss. we can do the same thing by hand  with simple math:

`output_loss = tf.reduce_sum(per_example_loss)*1./GLOBAL_BATCH_SIZE`

but tensorflow 2.0 has build in method to do this

#TODO: CHANGE TO ATTENTION
3. **Distributed train/test steps**  

Train and test step are almost the same so I`ll get into train step and point different in test step
Because we`re creating custom training loop there are two things we can use to speed up computations:

- use `@tf.function` to use static graph computation. We have to use it only in one method, because every method called inside it will automatically using it. Additionally it will speed up calcualtions, due to optimalization tensorflow makes when it uses it.  **to debug code please remove it, use tf.print(...) not normal python print(...)**
- training step uses `tf.GradientTape()` to keep track of gradients and allow backpropagation.

train_step(...) method makes one forward pass of training + applys gradients.
distributed_train_step(...) makes distributional part happens:

- we have to use `tf.strategy.MirroredStrategy.experimental_run_v2(method_name, args=(... ,))`   <- **IMPORTANT COMA AT THE END**
  to get distributed losses(vector of losses)
- then `tf.strategy.MirroredStrategy.reduce(tf.distribute.ReduceOp.SUM, ...)` to take sum of losses and calcualte output loss from whole distributed models

```python
            # one training step
            def train_step(encoder_input, decoder_in, decoder_out, initial_states):
                with tf.GradientTape() as tape:
                    encoder_states = self.encoder(encoder_input, initial_state, training=True)
                    predicted_data, _, _ = self.decoder(decoder_in, encoder_states[1:], training=True)
                    loss = compute_loss(predicted_data, decoder_out)

                trainable = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape.gradient(loss, trainable)
                self.optimizer.apply_gradients(zip(grads, trainable))
                train_accuracy.update_state(decoder_out, predicted_data)
                return loss

            @tf.function
            def distributed_train_step(encoder_input, decoder_in, decoder_out, initial_states):
                per_replica_losses = self.strategy.experimental_run_v2(train_step,
                                                              args=(encoder_input,
                                                                    decoder_in,
                                                                    decoder_out,
                                                                    initial_states,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)
```

Test_step differences:

- there is no `tf.GradientTape` + bakcpropagation step
- `training=False` for encoder/decoder object

```python
            def test_step(encoder_input, decoder_in, decoder_out):
                initial_state = self.encoder.init_states(self.batch_size)
                encoder_states = self.encoder(encoder_input, initial_state, training=False)
                predicted_data, _, _ = self.decoder(decoder_in, encoder_states[1:], training=False)
                loss = compute_loss(predicted_data, decoder_out)

                test_accuracy.update_state(decoder_out, predicted_data)
                return loss

            @tf.function
            def distributed_test_step(encoder_input, decoder_in, decoder_out):
                per_replica_losses = self.strategy.experimental_run_v2(test_step,
                                                              args=(encoder_input,
                                                                    decoder_in,
                                                                    decoder_out,))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)
```

4. **Prediction step**

It`s forward pass but feeded with <start> token at the beginnning.
Next steps are basically:

- take decoder output
- if it's `<end>` token break prediction loop
- else feed output of decoder last step as decoder input and repeat 

```python
            def predict(input_data, real_data_out):
                en_sentence = self.en_tokenizer.sequences_to_texts([input_data])
                input_data = tf.expand_dims(input_data, 0)
                initial_states = self.encoder.init_states(1)
                _, state_h, state_c = self.encoder(tf.constant(input_data), initial_states, training=False)

                symbol = tf.constant([[self.fr_tokenizer.word_index['<start>']]])
                sentence = []

                while True:
                    symbol, state_h, state_c = self.decoder(symbol, (state_h, state_c), training=False)
                    # argmax to get max index 
                    symbol = tf.argmax(symbol, axis=-1)
                    word = self.fr_tokenizer.index_word[symbol.numpy()[0][0]]

                    if word == '<end>' or len(sentence) >= len(real_data_out):
                        break

                    sentence.append(word)
                print("--------------PREDICTION--------------")
                print("  English   :  {}" .format(en_sentence))
                print("  Predicted :  {}" .format(' '.join(sentence)))
                print("  Correct   :  {}" .format(self.fr_tokenizer.sequences_to_texts([real_data_out])))
                print("------------END PREDICTION------------")
```

5. **Main loop**

Every eopch take proper batches and train/test. Additionally accuracy + losses are printed each x interations and losses/accuracy values are added to proper lists, so we can plot them after training finish.

```python
           for epoch in range(epochs):
                test_accuracy.reset_states()
                train_accuracy.reset_states()
                initial_state = self.encoder.init_states(self.batch_size)
                total_loss = 0.0
                num_batches = 0
                for _, (en_data, fr_data_in, fr_data_out) in enumerate(train_dataset_distr):
                    loss = distributed_train_step(en_data, fr_data_in, fr_data_out, initial_state)
                    total_loss += loss
                    num_batches += 1
                train_losses.append(total_loss/num_batches)
                total_loss = 0.0
                num_batches = 0
                for _, (en_data, fr_data_in, fr_data_out) in enumerate(test_dataset_distr):
                    loss = distributed_test_step(en_data, fr_data_in, fr_data_out)
                    total_loss += loss
                    num_batches += 1
                test_losses.append(total_loss/num_batches)
                print ('Epoch {} training Loss {:.4f} Accuracy {:.4f}  test Loss {:.4f} Accuracy {:.4f}' .format(
                                                      epoch + 1, 
                                                      train_losses[-1], 
                                                      train_accuracy.result(),
                                                      test_losses[-1],
                                                      test_accuracy.result()))
```

6. **Saving checkpoint**

One more thing that's going on here is saving model/optimizer value each x interations. It's done with `tf.train.Checkpoint`. For details please see [tensorflow site](https://www.tensorflow.org/guide/checkpoint)

Whole training process code can be found [here](https://github.com/mizzmir/NLP/blob/master/machine%20translation%20projects/Seq2SeqAttention/Seq2SeqAttentionTrainer.py)

7. **Results**

Accuracy plot:

Loss plot:

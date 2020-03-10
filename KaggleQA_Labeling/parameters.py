# We want to predict all needed answers in range 0 - 1 so 30 answers in range 0 -1 
num_labels = 30
epochs = 2
batch_size=4
lr = 3e-3
decay = 0.01
warmup_steps = 200
gradient_accumulate_steps=2
n_splits = 5
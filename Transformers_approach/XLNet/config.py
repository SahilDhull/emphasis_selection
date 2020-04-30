# Defines all paths and model hyperparameters

# Dataset files' paths
train_file = '../../datasets/train.txt'
dev_file = '../../datasets/dev.txt'
test_file = '../../datasets/test.txt'

# model name
model_name = 'xlnet-large-cased'

# Model hyperparameters

# Optimizer parameters
learning_rate = 2e-5
epsilon = 1e-8


num_epochs = 20

max_accuracy = 0
max_match = [0,0,0,0]

val_out = ""
test_out = ""

save_path = '../ensemble/'
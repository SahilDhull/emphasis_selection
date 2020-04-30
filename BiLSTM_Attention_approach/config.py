#Location of datasets
train_file = '../datasets/train.txt'
dev_file = '../datasets/dev.txt'

# Location of Glove embedding file, couldn't include in git due to its size
emb_file = 'glove/glove.6B.100d.txt'

#index of words, tags and probability in the input
word_index = 1 
tag_index = 5 
prob_index = 4 

# word and character should atleast appear once for inclusion in map
min_word_freq=1
min_char_freq=1

caseless=True #take all words caseless

#batch size and workers
batch_size = 10 
workers = 1 

expand_vocab = False #consider only the words already in the vocab

#elmo files
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


# Different hyperparameters
word_emb_dim = 2048 
word_rnn_dim = 512
char_rnn_dim = 300
dropout = 0.3
fine_tune_word_embeddings = False
target_size = 1
hidden_size = 20
char_emb_dim = 30
char_rnn_layers = 2
word_rnn_layers = 2

#learning rate
learning_rate = 0.0001
#print frequency
print_frequency = 25
epochs = 100 #epochs
# Initial Score values
max_score = 0 
max_m_scores = []





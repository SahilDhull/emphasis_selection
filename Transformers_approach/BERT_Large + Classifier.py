

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertConfig , BertForMaskedLM , BertModel
# from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizer
# from transformers import XLNetConfig, XLNetModel , XLNetTokenizer
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from transformers import PreTrainedModel, PreTrainedTokenizer , BertPreTrainedModel

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import codecs
from torch.nn.utils.rnn import pack_padded_sequence
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# from google.colab import drive
# drive.mount('/content/drive')

train_file = 'drive/My Drive/datasets/train.txt'
dev_file = 'drive/My Drive/datasets/dev.txt'
test_file = 'drive/My Drive/datasets/test.txt'

tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case = False)

def read_token_map(file,word_index = 1,prob_index = 4, caseless = False):
  
  with codecs.open(file, 'r', 'utf-8') as f:
      lines = f.readlines()

  tokenized_texts = []
  token_map = []
  token_labels = []
  sent_length = []

  bert_tokens = []
  orig_to_tok_map = []
  labels = []

  bert_tokens.append("[CLS]")
  
  for line in lines:
    if not (line.isspace()):
      feats = line.strip().split()
      word = feats[word_index].lower() if caseless else feats[word_index]
      label = feats[prob_index].lower() if caseless else feats[prob_index]
      labels.append((float)(label))
      orig_to_tok_map.append(len(bert_tokens))
      
      if(word == "n't"):
        word = "'t"
        if(bert_tokens[-1] != "won"):
          bert_tokens[-1] = bert_tokens[-1] +"n"
      if(word == "wo"):
        word = "won"

      bert_tokens.extend(tokenizer.tokenize(word))
     
    elif len(orig_to_tok_map) > 0:
      bert_tokens.append("[SEP]")
      tokenized_texts.append(bert_tokens)
      token_map.append(orig_to_tok_map)
      token_labels.append(labels)
      sent_length.append(len(labels))
      bert_tokens = []
      orig_to_tok_map = []
      labels = []
      length = 0
      bert_tokens.append("[CLS]")
          
  if len(orig_to_tok_map) > 0:
    bert_tokens.append("[SEP]")
    tokenized_texts.append(bert_tokens)
    token_map.append(orig_to_tok_map)
    token_labels.append(labels)
    sent_length.append(len(labels))
  
  return tokenized_texts, token_map, token_labels, sent_length

def read_test_token_map(file,word_index = 1, caseless = False):
  
  with codecs.open(file, 'r', 'utf-8') as f:
      lines = f.readlines()

  tokenized_texts = []
  token_map = []
  sent_length = []

  bert_tokens = []
  orig_to_tok_map = []
  
  bert_tokens.append("[CLS]")
  
  for line in lines:
    if not (line.isspace()):
      feats = line.strip().split()
      word = feats[word_index].lower() if caseless else feats[word_index]
      orig_to_tok_map.append(len(bert_tokens))
      
      if(word == "n't"):
        word = "'t"
        if(bert_tokens[-1] != "won"):
          bert_tokens[-1] = bert_tokens[-1] +"n"
      if(word == "wo"):
        word = "won"

      bert_tokens.extend(tokenizer.tokenize(word))
     
    elif len(orig_to_tok_map) > 0:
      bert_tokens.append("[SEP]")
      tokenized_texts.append(bert_tokens)
      token_map.append(orig_to_tok_map)
      sent_length.append(len(orig_to_tok_map))
      bert_tokens = []
      orig_to_tok_map = []
      length = 0
      bert_tokens.append("[CLS]")
          
  if len(orig_to_tok_map) > 0:
    bert_tokens.append("[SEP]")
    tokenized_texts.append(bert_tokens)
    token_map.append(orig_to_tok_map)
    sent_length.append(len(orig_to_tok_map))
  
  return tokenized_texts, token_map, sent_length

t_tokenized_texts, t_token_map, t_token_label, t_sent_length = read_token_map(train_file)
print(t_tokenized_texts[100])
print(t_token_map[100])
print(t_token_label[100])
print(t_sent_length[100])

d_tokenized_texts, d_token_map, d_token_label, d_sent_length = read_token_map(dev_file)
print(d_tokenized_texts[0])
print(d_token_map[0])
print(d_token_label[0])
print(d_sent_length[0])

f_tokenized_texts, f_token_map, f_sent_length = read_test_token_map(test_file)
print(f_tokenized_texts[50])
print(f_token_map[50])
print(f_sent_length[50])

MAX_LEN = 72

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
t_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in t_tokenized_texts]

# Pad our input tokens
t_input_ids = pad_sequences(t_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
t_token_map = pad_sequences(t_token_map, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
t_token_label = pad_sequences(t_token_label, maxlen=MAX_LEN, dtype="float", truncating="post", padding="post")

print(t_input_ids[100])
print(t_token_map[100])
print(t_token_label[100])

d_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in d_tokenized_texts]

# Pad our input tokens
d_input_ids = pad_sequences(d_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
d_token_map = pad_sequences(d_token_map, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
d_token_label = pad_sequences(d_token_label, maxlen=MAX_LEN, dtype="float", truncating="post", padding="post")

print(d_input_ids[0])
print(d_token_map[0])
print(d_token_label[0])

f_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in f_tokenized_texts]

# Pad our input tokens
f_input_ids = pad_sequences(f_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
f_token_map = pad_sequences(f_token_map, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

print(f_input_ids[50])
print(f_token_map[50])

t_attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in t_input_ids:
  seq_mask = [float(i>0) for i in seq]
  t_attention_masks.append(seq_mask)
print(t_attention_masks[100])

d_attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in d_input_ids:
  seq_mask = [float(i>0) for i in seq]
  d_attention_masks.append(seq_mask)
print(d_attention_masks[0])

f_attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in f_input_ids:
  seq_mask = [float(i>0) for i in seq]
  f_attention_masks.append(seq_mask)
print(f_attention_masks[50])

t_input_ids = torch.tensor(t_input_ids)
t_token_map = torch.tensor(t_token_map )
t_token_label = torch.tensor(t_token_label)
t_attention_masks = torch.tensor(t_attention_masks)
t_sent_length = torch.tensor(t_sent_length)

d_input_ids = torch.tensor(d_input_ids)
d_token_map = torch.tensor(d_token_map )
d_token_label = torch.tensor(d_token_label)
d_attention_masks = torch.tensor(d_attention_masks)
d_sent_length = torch.tensor(d_sent_length)

f_input_ids = torch.tensor(f_input_ids)
f_token_map = torch.tensor(f_token_map )
f_attention_masks = torch.tensor(f_attention_masks)
f_sent_length = torch.tensor(f_sent_length)

# Select a batch size for training. 
batch_size = 32
# print(t_token_labels)
# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(t_input_ids, t_token_map, t_token_label, t_attention_masks, t_sent_length)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(d_input_ids, d_token_map, d_token_label, d_attention_masks, d_sent_length)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size, shuffle = False)
test_data = TensorDataset(f_input_ids, f_token_map, f_attention_masks, f_sent_length)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size,shuffle = False)

def read_for_output(file,word_index = 1):
  
  with codecs.open(file, 'r', 'utf-8') as f:
      lines = f.readlines()

  words_lsts = []
  word_ids_lsts = []
  words = []
  ids = []
  
  for line in lines:
    if not (line.isspace()):
      feats = line.strip().split()
      words.append(feats[word_index])
      ids.append(feats[0])
     
    elif len(words) > 0:
      words_lsts.append(words)
      word_ids_lsts.append(ids)
      words = []
      ids = []
          
  if len(words) > 0:
    words_lsts.append(words)
    word_ids_lsts.append(ids)
    words = []
    ids = []
  
  return words_lsts , word_ids_lsts

dev_words, dev_word_ids = read_for_output(dev_file)
test_words, test_word_ids = read_for_output(test_file)

print(dev_words[0])
print(dev_word_ids[0])
print(test_words[50])
print(test_word_ids[50])

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def fix_padding(scores_numpy, label_probs,  mask_numpy):
    #if len(scores_numpy) != len(mask_numpy):
    #    print("Error: len(scores_numpy) != len(mask_numpy)")
    #assert len(scores_numpy) == len(mask_numpy)
    #if len(label_probs) != len(mask_numpy):
    #    print("len(label_probs) != len(mask_numpy)")
    #assert len(label_probs) == len(mask_numpy)

    all_scores_no_padd = []
    all_labels_no_pad = []
    for i in range(len(mask_numpy)):
        all_scores_no_padd.append(scores_numpy[i][:int(mask_numpy[i])])
        all_labels_no_pad.append(label_probs[i][:int(mask_numpy[i])])

    assert len(all_scores_no_padd) == len(all_labels_no_pad)
    return all_scores_no_padd, all_labels_no_pad

def match_M(batch_scores_no_padd, batch_labels_no_pad):

    top_m = [1, 2, 3, 4]
    batch_num_m=[]
    batch_score_m=[]
    for m in top_m:
        intersects_lst = []
        # exact_lst = []
        score_lst = []
        ############################################### computing scores:
        for s in batch_scores_no_padd:
            if len(s) <=m:
                continue
            h = m
            # if len(s) > h:
            #     while (s[np.argsort(s)[-h]] == s[np.argsort(s)[-(h + 1)]] and h < (len(s) - 1)):
            #         h += 1

            # s = np.asarray(s.cpu())
            s = np.asarray(s)
            #ind_score = np.argsort(s)[-h:]
            ind_score = sorted(range(len(s)), key = lambda sub: s[sub])[-h:]
            score_lst.append(ind_score)

        ############################################### computing labels:
        label_lst = []
        for l in batch_labels_no_pad:
            if len(l) <=m:
                continue
            # if it contains several top values with the same amount
            h = m
            # l = l.cpu()
            if len(l) > h:
                while (l[np.argsort(l)[-h]] == l[np.argsort(l)[-(h + 1)]] and h < (len(l) - 1)):
                    h += 1
            l = np.asarray(l)
            ind_label = np.argsort(l)[-h:]
            label_lst.append(ind_label)

        ############################################### :

        for i in range(len(score_lst)):
            intersect = intersection(score_lst[i], label_lst[i])
            intersects_lst.append((len(intersect))/(min(m, len(score_lst[i]))))
            # sorted_score_lst = sorted(score_lst[i])
            # sorted_label_lst =  sorted(label_lst[i])
            # if sorted_score_lst==sorted_label_lst:
            #     exact_lst.append(1)
            # else:
            #     exact_lst.append(0)
        batch_num_m.append(len(score_lst))
        batch_score_m.append(sum(intersects_lst))
    return batch_num_m, batch_score_m

def test(model):
  print("")
  print("Running test...")

  model.eval()
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  iii = 0

  s = ""
  sentence_id = ""

  for batch in test_dataloader:
      
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      
      # Unpack the inputs from our dataloader
      v_input_ids = batch[0].to(device)
      v_input_mask = batch[2].to(device)
      v_token_starts = batch[1].to(device)
      v_sent_length = batch[3]
            
      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():        
          output = model(v_input_ids, v_input_mask, v_token_starts, v_sent_length)
      
      pred_labels = output[1]

      pred_labels = pred_labels.detach().cpu().numpy()

      for i in range(v_input_ids.size()[0]):
        for j in range(len(test_words[iii])):
          if sentence_id == iii:
            s = s + "{}\t{}\t{}\t".format(test_word_ids[iii][j], test_words[iii][j], pred_labels[i][j]) + "\n"
          else:
            s = s + "\n" + "{}\t{}\t{}\t".format(test_word_ids[iii][j], test_words[iii][j], pred_labels[i][j]) + "\n"
            sentence_id = iii
        iii = iii + 1
      s = s +"\n"
      
  print("testing complete\n")
  # print(s)
  return s

def validation(model):
  print("")
  print("Running Validation...")

  model.eval()
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  num_m = [0, 0, 0, 0]
  score_m = [0, 0, 0, 0]

  iii = 0

  s = ""
  sentence_id = ""

  for batch in validation_dataloader:
      
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      
      # Unpack the inputs from our dataloader
      v_input_ids = batch[0].to(device)
      v_input_mask = batch[3].to(device)
      v_token_starts = batch[1].to(device)
      v_labels = batch[2].to(device)
      v_sent_length = batch[4]
            
      # Telling the model not to compute or store gradients, saving memory and
      # speeding up validation
      with torch.no_grad():        
          output = model(v_input_ids, v_input_mask, v_token_starts, v_sent_length, v_labels)
      
      pred_labels = output[1]

      pred_labels = pred_labels.detach().cpu().numpy()
      v_labels = v_labels.to('cpu').numpy()

      for i in range(v_input_ids.size()[0]):
        for j in range(len(dev_words[iii])):
          if sentence_id == iii:
            s = s + "{}\t{}\t{}\t{}".format(dev_word_ids[iii][j], dev_words[iii][j], pred_labels[i][j],v_labels[i][j]) + "\n"
          else:
            s = s + "\n" + "{}\t{}\t{}\t{}".format(dev_word_ids[iii][j], dev_words[iii][j], pred_labels[i][j],v_labels[i][j]) + "\n"
            sentence_id = iii
        iii = iii + 1
      s = s +"\n"
      
      pred_labels, v_labels = fix_padding(pred_labels, v_labels, v_sent_length)

      batch_num_m, batch_score_m = match_M(pred_labels, v_labels)
      num_m = [sum(i) for i in zip(num_m, batch_num_m)]
      
      score_m = [sum(i) for i in zip(score_m, batch_score_m)]
  
  m_score = [i/j for i,j in zip(score_m, num_m)]
  
  # print(num_m)
  # print(score_m)
  print("Validation Accuracy: ")
  print(m_score)
  v_score = np.mean(m_score)
  print(v_score)
  # print(s)

  return v_score, m_score, s

max_accuracy = 0
max_array = []
val_out = ""
test_out = ""

def train(model,  optimizer, scheduler, tokenizer, max_epochs, save_path, device, val_freq = 10):
  
  bestpoint_dir = os.path.join(save_path)
  os.makedirs(bestpoint_dir, exist_ok=True)

  global max_accuracy 
  global max_array
  global val_out 
  global test_out 
  
  for epoch_i in range(0, max_epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, max_epochs))
    print('Training...')

    # Reset the total loss for this epoch.
    total_loss = 0
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):    

        print("batch",step,"out of",len(train_dataloader))
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[3].to(device)
        b_token_starts = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_sent_length = batch[4]

        model.zero_grad()   
        model.train()     

        output = model(b_input_ids, b_input_mask, b_token_starts,b_sent_length,b_labels)
        loss = output[0]

        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        if step % 10 == 0:
          accuracy, array, outs = validation(model)
          if(accuracy > max_accuracy):
            max_accuracy = accuracy
            max_array = array
            val_out = outs
            test_out = test(model)

            # model.save_pretrained(bestpoint_dir)  
            # print("Saving model bestpoint to ", bestpoint_dir)
          
          print(max_accuracy)
          print("")

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
  
  print("")
  print("Training complete!")

class transformer_model(nn.Module):
  def __init__(self, model_name, drop_prob = 0.3):
    super(transformer_model, self).__init__()

    config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    self.bert = BertForMaskedLM.from_pretrained(model_name, config = config)
    
    # the commented lines freezes layers of the model
    # cnt=0
    # for child in bert.bert.children():
    #   cnt = cnt + 1
    #   if cnt<=23:
    #     for param in child.parameters():
    #       param.requires_grad = False

    bert_dim = 24*1024
    hidden_dim1 = 900
    hidden_dim2 = 40
    final_size = 1

    self.fc1 = nn.Linear(bert_dim, hidden_dim1)
    self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
    self.fc3 = nn.Linear(hidden_dim2, final_size)
    self.dropout = nn.Dropout(p=drop_prob)

  def avg(self, a, st, end):
    k = a
    lis = []
    for i in range(st,end):
      lis.append(a[i])
    x = torch.mean(torch.stack(lis),dim=0)
    return x

  def save_pretrained(self, output_dir):
    self.bert.save_pretrained(output_dir)
    #please save the fc layers
           
  def forward(self, bert_ids, bert_mask, bert_token_starts, lm_lengths = None, labels = None):
    
    batch_size = bert_ids.size()[0]
    pad_size = bert_ids.size()[1]
    # print("batch size",batch_size,"\t\tpad_size",pad_size)

    output = self.bert(bert_ids, attention_mask = bert_mask)

    bert_out = output[-1][0]
    for layers in range(1,24,1):
      bert_out = torch.cat((bert_out, output[-1][layers]), dim=2)
    
    pred_logits = torch.relu(self.fc1(self.dropout(bert_out)))
    pred_logits = torch.relu(self.fc2(self.dropout(pred_logits)))
    pred_logits = torch.sigmoid(self.fc3(self.dropout(pred_logits)))
    pred_logits = torch.squeeze(pred_logits,2)

    pred_labels = torch.tensor(np.zeros(bert_token_starts.size()),dtype = torch.float64).to(device)

    # print(pred_logits[0])
    # print(pred_labels[0])
    # print(labels[0])
    # print(bert_token_starts[0])

    for b in range(batch_size):
      for w in range(pad_size):
        if(bert_token_starts[b][w]!=0):
          if(bert_token_starts[b][w]>=pad_size):
            print(bert_token_starts[b])
          else:
            st = bert_token_starts[b][w]
            end = bert_token_starts[b][w+1]
            if(end==0):
              end = st+1
              while(bert_mask[b][end]!=0):
                end = end+1
            # pred_labels[b][w] = self.avg(pred_logits[b],st,end)
            pred_labels[b][w] = pred_logits[b][bert_token_starts[b][w]]

    # print(pred_labels[0])

    if(labels != None):
      lm_lengths, lm_sort_ind = lm_lengths.sort(dim=0, descending=True)
      scores = labels[lm_sort_ind]
      targets = pred_labels[lm_sort_ind]
      scores = pack_padded_sequence(scores, lm_lengths, batch_first=True).data
      targets = pack_padded_sequence(targets, lm_lengths, batch_first=True).data

      # print(targets,scores)

      loss_fn = nn.BCELoss().to(device) 
      loss = loss_fn(targets,scores)

      return loss, pred_labels 

    else:
      return 0.0, pred_labels

model = transformer_model('bert-large-cased').to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps = 1e-8)

epochs = 10
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

save_path = 'drive/My Drive/datasets/ensemble/bert_maskedLM/'
train(model,  optimizer, scheduler, tokenizer, epochs, save_path, device)

print(max_accuracy)
print(max_array)

save_path = 'drive/My Drive/datasets/ensemble/bert_maskedLM/'
def print_to_file(file_path, var):
  with open(save_path + file_path, "w") as text_file:
    text_file.write(var)

ind = 6
print_to_file('val'+str(ind)+'.txt',val_out)
print_to_file('test'+str(ind)+'.txt',test_out)
print_to_file('max'+str(ind)+'.txt', str(max_accuracy)+"\n"+str(max_array))

# print(val_out, test_out)
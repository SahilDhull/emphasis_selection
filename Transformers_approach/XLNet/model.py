from config import *

class transformer_model(nn.Module):
  def __init__(self, model_name, drop_prob = dropout_prob):
    super(transformer_model, self).__init__()

    configuration = XLNetConfig.from_pretrained(model_name, output_hidden_states=True)
    self.xlnet = XLNetModel.from_pretrained(model_name, config = configuration)
    
    # freezes layers of the model
    if to_freeze:
      cnt=0
      for child in xlnet.xlnet.children():
        cnt = cnt + 1
        if cnt<=freeze_layers:
          for param in child.parameters():
            param.requires_grad = False

    self.fc1 = nn.Linear(xlnet_dim, hidden_dim1)
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
        
  def forward(self, xlnet_ids, xlnet_mask, xlnet_token_starts, lm_lengths = None, labels = None):
    
    batch_size = xlnet_ids.size()[0]
    pad_size = xlnet_ids.size()[1]
    # print("batch size",batch_size,"\t\tpad_size",pad_size)

    output = self.xlnet(xlnet_ids, attention_mask = xlnet_mask)

    # Concatenating hidden dimensions of all encoder layers
    xlnet_out = output[-1][0]
    for layers in range(1,25,1):
      xlnet_out = torch.cat((xlnet_out, output[-1][layers]), dim=2)
    
    # Fully connected layers with relu and dropouts in between
    pred_logits = torch.relu(self.fc1(self.dropout(xlnet_out)))
    pred_logits = torch.relu(self.fc2(self.dropout(pred_logits)))
    pred_logits = torch.sigmoid(self.fc3(self.dropout(pred_logits)))
    pred_logits = torch.squeeze(pred_logits,2)

    pred_labels = torch.tensor(np.zeros(xlnet_token_starts.size()),dtype = torch.float64).to(device)

    for b in range(batch_size):
      for w in range(pad_size):
        if(xlnet_token_starts[b][w]!=0):
          if(xlnet_token_starts[b][w]>=pad_size):
            print(xlnet_token_starts[b])
          else:
            st = xlnet_token_starts[b][w]
            end = xlnet_token_starts[b][w+1]
            if(end==0):
              end = st+1
              while(xlnet_mask[b][end]!=0):
                end = end+1
            # For using average or just the first token of a word (in case of word splitting by tokenizer)
            # pred_labels[b][w] = self.avg(pred_logits[b],st,end)
            pred_labels[b][w] = pred_logits[b][xlnet_token_starts[b][w]]


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
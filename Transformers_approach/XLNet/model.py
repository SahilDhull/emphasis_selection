class transformer_model(nn.Module):
  def __init__(self, model_name, drop_prob = 0.3):
    super(transformer_model, self).__init__()

    config = XLNetConfig.from_pretrained('xlnet-large-cased', output_hidden_states=True)
    self.xlnet = XLNetModel.from_pretrained('xlnet-large-cased', config = config)
    
    # the commented lines freezes layers of the model
    # cnt=0
    # for child in xlnet.xlnet.children():
    #   cnt = cnt + 1
    #   if cnt<=23:
    #     for param in child.parameters():
    #       param.requires_grad = False

    xlnet_dim = 25*1024
    hidden_dim1 = 1000
    hidden_dim2 = 40
    final_size = 1

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

    xlnet_out = output[-1][0]
    for layers in range(1,25,1):
      xlnet_out = torch.cat((xlnet_out, output[-1][layers]), dim=2)
    
    pred_logits = torch.relu(self.fc1(self.dropout(xlnet_out)))
    pred_logits = torch.relu(self.fc2(self.dropout(pred_logits)))
    pred_logits = torch.sigmoid(self.fc3(self.dropout(pred_logits)))
    pred_logits = torch.squeeze(pred_logits,2)

    pred_labels = torch.tensor(np.zeros(xlnet_token_starts.size()),dtype = torch.float64).to(device)

    # print(pred_logits[0])
    # print(pred_labels[0])
    # print(labels[0])
    # print(xlnet_token_starts[0])

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
            # pred_labels[b][w] = self.avg(pred_logits[b],st,end)
            pred_labels[b][w] = pred_logits[b][xlnet_token_starts[b][w]]

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
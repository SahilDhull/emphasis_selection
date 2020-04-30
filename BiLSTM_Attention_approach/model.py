from config import *

class ElmoLayer(nn.Module):
    def __init__(self,options_file, weight_file):
        super(ElmoLayer, self).__init__()
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0.3) #defining Elmo with 2 num_output_representations and 0.3 dropout

    def forward(self, words):
        character_ids = batch_to_ids(words) #converting words to ids
        character_ids = character_ids.to(device)
        elmo_output = self.elmo(character_ids)
        elmo_representation = torch.cat(elmo_output['elmo_representations'], -1)
        elmo_representation.to(device)
        if torch.cuda.is_available():
            elmo_representation = elmo_representation.cuda()
        return elmo_representation 
    

class Attention(nn.Module):
    """Attention mechanism written by Gustavo Aguilar https://github.com/gaguilar"""
    def __init__(self,  hidden_size):
        super(Attention, self).__init__()
        self.da = hidden_size
        self.dh = hidden_size

        self.W = nn.Linear(self.dh, self.da)        # (feat_dim, attn_dim)
        self.v = nn.Linear(self.da, 1)              # (attn_dim, 1)

    def forward(self, inputs, mask):
        # Raw scores
        u = self.v(torch.tanh(self.W(inputs)))      # (batch, seq, hidden) -> (batch, seq, attn) -> (batch, seq, 1)

        # Masked softmax
        u = u.exp()                                 # exp to calculate softmax
        u = mask.unsqueeze(2).float().to(device) * u           # (batch, seq, 1) * (batch, seq, 1) to zerout out-of-mask numbers
        sums = torch.sum(u, dim=1, keepdim=True)    # now we are sure only in-mask values are in sum
        a = u / sums                                # the probability distribution only goes to in-mask values now

        # Weighted vectors
        z = inputs * a

        return  z,  a.view(inputs.size(0), inputs.size(1))


class Highway(nn.Module):
    """
    Highway Network.
    """

    def __init__(self, size, num_layers=1, dropout=0.5):
        """
        :param size: size of linear layer (matches input size)
        :param num_layers: number of transform and gate layers
        :param dropout: dropout
        """
        super(Highway, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.transform = nn.ModuleList()  # list of transform layers
        self.gate = nn.ModuleList()  # list of gate layers
        self.dropout = nn.Dropout(p=dropout)

        for i in range(num_layers):
            transform = nn.Linear(size, size)
            gate = nn.Linear(size, size)
            self.transform.append(transform)
            self.gate.append(gate)

    def forward(self, x):
        """
        Forward propagation.
        :param x: input tensor
        :return: output tensor, with same dimensions as input tensor
        """
        transformed = nn.functional.relu(self.transform[0](x))  # transform input
        g = nn.functional.sigmoid(self.gate[0](x))  # calculate how much of the transformed input to keep

        out = g * transformed + (1 - g) * x  # combine input and transformed input in this ratio

        # If there are additional layers
        for i in range(1, self.num_layers):
            out = self.dropout(out)
            transformed = nn.functional.relu(self.transform[i](out))
            g = nn.functional.sigmoid(self.gate[i](out))

            out = g * transformed + (1 - g) * out

        return out


class LM_LSTM_CRF(nn.Module):

    def __init__(self, charset_size, char_emb_dim, char_rnn_dim, char_rnn_layers,
                 lm_vocab_size, word_emb_dim, word_rnn_dim, word_rnn_layers, dropout, highway_layers=1):
        
        super(LM_LSTM_CRF, self).__init__()

        self.target_size = target_size  # this is the size of the output vocab of the tagging model
        self.hidden_size = hidden_size
        
        self.charset_size = charset_size
        self.char_emb_dim = char_emb_dim
        self.char_rnn_dim = char_rnn_dim
        self.char_rnn_layers = char_rnn_layers
        

        self.wordset_size = lm_vocab_size  # this is the size of the input vocab (embedding layer) of the tagging model
        self.word_emb_dim = word_emb_dim
        self.word_rnn_dim = word_rnn_dim
        self.word_rnn_layers = word_rnn_layers
        
        self.highway_layers = highway_layers

        self.dropout = nn.Dropout(p=dropout)
        
        self.char_embeds = nn.Embedding(self.charset_size, self.char_emb_dim)  # character embedding layer
        self.forw_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, dropout=dropout)  # forward character LSTM
        self.back_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, dropout=dropout)  # backward character LSTM

        self.subword_hw = Highway(2 * self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout).to(device)  # highway to transform combined forward and backward char LSTM outputs for use in the word BLSTM

        self.word_embeds = nn.Embedding(self.wordset_size, self.word_emb_dim)  # word embedding layer
        self.elmo = ElmoLayer(options_file, weight_file)
        self.word_blstm = nn.LSTM(self.word_emb_dim + self.char_rnn_dim * 2, self.word_rnn_dim, num_layers=self.word_rnn_layers, bidirectional=True, dropout=dropout)  # word BLSTM
        


        self.attention = Attention(self.word_rnn_dim*2).to(device)
        
        self.fc1 = nn.Linear((self.word_rnn_dim*2)+1, self.hidden_size) #Note that the extra +1 is for the pos tag concat
        self.fc2 = nn.Linear(self.hidden_size, self.target_size)
        
    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.word_embeds.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).
        :param fine_tune: Fine-tune?
        """
        for p in self.word_embeds.parameters():
            p.requires_grad = fine_tune

    def forward(self, words, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, wmap_lengths, cmap_lengths, probs, tmaps):
        
        self.batch_size = cmaps_f.size(0)
        self.word_pad_len = wmaps.size(1)

        # Sort by decreasing true char. sequence length
        cmap_lengths, char_sort_ind = cmap_lengths.sort(dim=0, descending=True)
        cmaps_f = cmaps_f[char_sort_ind]
        cmaps_b = cmaps_b[char_sort_ind]
        cmarkers_f = cmarkers_f[char_sort_ind]
        cmarkers_b = cmarkers_b[char_sort_ind]
        wmaps = wmaps[char_sort_ind]
        tmaps = tmaps[char_sort_ind]
        wmap_lengths = wmap_lengths[char_sort_ind]
        probs = probs[char_sort_ind]
        char_order = char_sort_ind.tolist()
        words_char_sorted = []
        for i in range(len(char_order)):
            words_char_sorted.append(words[char_order[i]])
        # Embedding look-up for characters
        cf = self.char_embeds(cmaps_f.to(device))  # (batch_size, char_pad_len, char_emb_dim)
        cb = self.char_embeds(cmaps_b.to(device))

        # Dropout
        cf = self.dropout(cf)  # (batch_size, char_pad_len, char_emb_dim)
        cb = self.dropout(cb)

        # Pack padded sequence
        cf = pack_padded_sequence(cf, cmap_lengths.tolist(),
                                  batch_first=True)  # packed sequence of char_emb_dim, with real sequence lengths
        cb = pack_padded_sequence(cb, cmap_lengths.tolist(), batch_first=True)

        # LSTM
        cf, _ = self.forw_char_lstm(cf)  # packed sequence of char_rnn_dim, with real sequence lengths
        cb, _ = self.back_char_lstm(cb)

        # Unpack packed sequence
        cf, _ = pad_packed_sequence(cf, batch_first=True)  # (batch_size, max_char_len_in_batch, char_rnn_dim)
        cb, _ = pad_packed_sequence(cb, batch_first=True)

        # Sanity check
        assert cf.size(1) == max(cmap_lengths.tolist()) == list(cmap_lengths)[0]
        
        
        # Select RNN outputs only at marker points (spaces in the character sequence)
        cmarkers_f = cmarkers_f.unsqueeze(2).expand(self.batch_size, self.word_pad_len, self.char_rnn_dim)
        cmarkers_b = cmarkers_b.unsqueeze(2).expand(self.batch_size, self.word_pad_len, self.char_rnn_dim)
        cf_selected = torch.gather(cf, 1, cmarkers_f)  # (batch_size, word_pad_len, char_rnn_dim)
        cb_selected = torch.gather(cb, 1, cmarkers_b)


        # Sort by decreasing true word sequence length
        wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)
        wmaps = wmaps[word_sort_ind]
        probs = probs[word_sort_ind]
        tmaps = tmaps[word_sort_ind]
        cf_selected = cf_selected[word_sort_ind]  
        cb_selected = cb_selected[word_sort_ind]
        word_order = word_sort_ind.tolist()
        words_word_sorted = []
        for i in range(len(word_order)):
            words_word_sorted.append(words_char_sorted[word_order[i]])
        w = self.elmo(words_word_sorted)
        
        
        # Sub-word information at each word
        subword = self.subword_hw(self.dropout(
            torch.cat((cf_selected, cb_selected), dim=2)))  # (batch_size, word_pad_len, 2 * char_rnn_dim)
        
        
        subword = self.dropout(subword)
        
        # Concatenate word embeddings and sub-word features
        w = torch.cat((w, subword), dim=2)  # (batch_size, word_pad_len, word_emb_dim + 2 * char_rnn_dim)

        # Pack padded sequence
        w = pack_padded_sequence(w, list(wmap_lengths), batch_first=True)  # packed sequence of word_emb_dim + 2 * char_rnn_dim, with real sequence lengths
        
        # LSTM
        w, _ = self.word_blstm(w)  # packed sequence of word_rnn_dim, with real sequence lengths

        # Unpack packed sequence
        w, _ = pad_packed_sequence(w, batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)
        w = self.dropout(w)
        
        mask = [] # creating mask 1 for the token where actual word is there and zero for padding
        for i in range(len(wmap_lengths)):
            mask_row = (np.concatenate([np.ones(wmap_lengths[i]),np.zeros(len(wmaps[i])-wmap_lengths[i])])).tolist()
            mask.append(mask_row)
        #Attention
        att_output, att_weights = self.attention(w, torch.from_numpy(np.asarray(mask)).float())
        
        tmaps = (tmaps.unsqueeze_(-1)).expand(list(att_output.size())[0],list(att_output.size())[1],1).float()
        att_output = torch.cat([att_output, tmaps],2)
        
        
        # fc layers 
        w = torch.relu(self.fc1(att_output)) 
        w = self.dropout(w)
        
        # final score
        scores = torch.sigmoid(self.fc2(w)) 
                
        return scores, tmaps, wmaps, probs, wmap_lengths, word_sort_ind

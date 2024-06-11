import torch
import torch.nn as nn
from utils.data_processing import *

class LSTM_CNN_Arch_With_Attention(nn.Module):
    def __init__(self, model_config):
        super(LSTM_CNN_Arch_With_Attention, self).__init__()
        num_enc_lstm_layers,embed_size,enc_input_size,enc_hidden_size,local_vocab_size,vocab_size,num_dec_lstm_layers,dec_hidden_size,is_use_cuda = model_config["num_enc_lstm_layers"],model_config["embed_size"],model_config["enc_input_size"],model_config["enc_hidden_size"],model_config["local_vocab_size"],model_config["vocab_size"],model_config["num_dec_lstm_layers"],model_config["dec_hidden_size"],model_config.get("is_use_cuda",True)
        if not is_use_cuda:
            self.device = "cpu"
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.local_vocab_size = local_vocab_size
        self.encoder = Enc_LSTM_CNN_Arch(num_enc_lstm_layers,embed_size,enc_input_size,enc_hidden_size,local_vocab_size,is_use_cuda)
        print("self.encoder",sum(p.numel() for p in self.encoder.parameters()))
        self.decoder = Dec_LSTM_CNN_Arch(num_dec_lstm_layers,embed_size,dec_hidden_size,local_vocab_size,is_use_cuda)
        print("self.decoder",sum(p.numel() for p in self.decoder.parameters()))
        self.attention_layer = nn.MultiheadAttention(embed_dim=dec_hidden_size, num_heads=4,kdim=2*enc_hidden_size, vdim=2*enc_hidden_size, batch_first=True, device=self.device)
        # AdaptiveLogSoftmaxWithLoss decreases the compute time drastically when vocab size is massive by forming clusters and doing kind of hierarchical clustering based softmax
        # The cutoffs are set seeing the frequency distribution graph of the W2V vocab words
        # The vocab size is global vocabulary whereas for encoder and decoder inputs vocab size is local vocabulary words(This is necessary bcoz embedding size would be massive otherwise)
        self.adapt_smax_layer = nn.AdaptiveLogSoftmaxWithLoss(2*dec_hidden_size, vocab_size,cutoffs=[5000*10,5000*25,5000*100,5000*350],div_value=2.0,device=self.device)
        print("self.adapt_smax_layer",sum(p.numel() for p in self.adapt_smax_layer.parameters()))
        # self.classification_layer = nn.Linear(2*enc_hidden_size, vocab_size,device=self.device) This doesn't work since vocab size is massive and this single operation becomes the bottleneck
        
    def forward(self, overall_inp,wordvec_obj_list=None,vectorizer_func=None,index_func=None,local_vocab_key_to_indx=None,overall_key_to_index=None,overall_index_to_key=None):
        enc_w2v_embed,dec_w2v_embed,enc_ind_embed,dec_ind_embed,labels_seqind=overall_inp
        cur_bs_size = enc_w2v_embed.size()[0]
        overall_loss = None
        # Key attention mask doesn't work for our architecture bcoz CNNs change the number of timesteps in the encoder
        # key_attention_mask = torch.zeros_like(enc_ind_embed,device=self.device,dtype=torch.bool)
        # print("key_attention_mask:{}".format(key_attention_mask.size()))
        # key_attention_mask = torch.where(enc_ind_embed == 0,True,key_attention_mask)
        enc_out,(_,_) = self.encoder(enc_w2v_embed,enc_ind_embed)
        # print("enc_out",enc_out.size())
        if dec_w2v_embed is not None:
            dec_out,(_,_) = self.decoder(dec_w2v_embed,dec_ind_embed)
            # print("dec_out",dec_out.size())
            attn_out,_ = self.attention_layer(query=dec_out, key=enc_out, value=enc_out, need_weights=False)
            attn_out = torch.cat((attn_out,dec_out),dim=2)
            # !!!! It is important to have timestep first dimension in the tensor bcoz the inner loop has N,B hence the memory requirements stay same and predictable as compared to having timestep in the second dimension which results in heavy memory footprint changes
            attn_out = torch.transpose(attn_out,0,1)
            if labels_seqind is not None:
                labels_seqind = torch.transpose(labels_seqind,0,1)
            output = []
            overall_loss = 0
            cnt = 0
            end_tag_mask = torch.ones(attn_out.size()[1],device=attn_out.device)
            for ind in range(attn_out.size()[0]):
                cur_out = self.adapt_smax_layer.predict(attn_out[ind])
                cur_mask = torch.where(cur_out == 2,0,1)
                cur_out = cur_out * end_tag_mask
                end_tag_mask = end_tag_mask * cur_mask
                output.append(cur_out)
                if labels_seqind is not None:
                    out,_ = self.adapt_smax_layer(attn_out[ind],labels_seqind[ind])
                    # Removing padding entries from tensor before loss calculation
                    out = out[labels_seqind[ind] != 0]
                    if(out.size()[0]>0):
                        cnt += 1
                        overall_loss += torch.mean(-out)
            overall_loss = (overall_loss / cnt)
            output = torch.stack(output)
            output = torch.transpose(output,0,1).long()
            if labels_seqind is not None:
                labels_seqind = torch.transpose(labels_seqind,0,1)
            # print("output:{} labels_seqind:{} overall_loss:{}".format(output.size(),labels_seqind.size(),overall_loss))
        else:
            # Run this in inference mode. Here we will decode one word at a time. So always output size per timestep is N
            cur_output,dec_h_out,dec_c_out = None,None,None
            output = []
            if wordvec_obj_list is None:
                wordvec_obj_list = word2vec_obj_list
            if vectorizer_func is None:
                vectorizer_func = sentence_vectorizer_using_wordembed
            if index_func is None:
                index_func = convert_tokens_to_indices
            is_done = torch.zeros((cur_bs_size),device=self.device).long()
            maxlen = 500
            while(torch.sum(is_done)<cur_bs_size and len(output)<maxlen):
                if(cur_output is None):
                    cur_output = [STRT] * cur_bs_size
                else:
                    cur_output = [overall_index_to_key[ind.item()] for ind in cur_output]
                dec_w2v_embed = torch.unsqueeze(torch.from_numpy(vectorizer_func(cur_output,wordvec_obj_list)),dim=1)
                dec_ind_embed = torch.unsqueeze(torch.tensor(index_func(cur_output,local_vocab_key_to_indx)),dim=1)
                # print("dec_w2v_embed:{} dec_ind_embed:{}".format(dec_w2v_embed.size(),dec_ind_embed.size()))

                dec_out,(dec_h_out,dec_c_out) = self.decoder(dec_w2v_embed,dec_ind_embed,dec_h_out,dec_c_out)
                attn_out,_ = self.attention_layer(query=dec_out, key=enc_out, value=enc_out, need_weights=False)
                attn_out = torch.squeeze(torch.cat((attn_out,dec_out),dim=2),dim=1)
                # print("attn_out :{}".format(attn_out.size()))
                cur_output = self.adapt_smax_layer.predict(attn_out)
                # If a sample is marked done, further sequence is replaced by PAD
                cur_output = torch.where(is_done == 1,0,cur_output)
                # When in a particular sample, END token is seen mark that sample as done
                is_done = torch.where(cur_output == overall_key_to_index[END],1,is_done)
                output.append(cur_output)
            
            # Important!! This is needed bcoz when we use multiple GPUs to run, during gather different size of dimension 1 gives error during stacking
            pad_seq = None
            if(len(output) < maxlen):
                pad_seq = torch.zeros((maxlen - len(output),cur_bs_size),dtype=output[0].dtype,device=output[0].device)
            output = torch.stack(output)
            if pad_seq is not None:
                output = torch.cat((output,pad_seq),dim=0)
            output = torch.transpose(output,0,1)

        return output,overall_loss       


class Enc_LSTM_CNN_Arch(nn.Module):
    def __init__(self, num_enc_lstm_layers,embed_size,enc_input_size,enc_hidden_size,vocab_size,is_use_cuda=True):
        super(Enc_LSTM_CNN_Arch, self).__init__()
        if not is_use_cuda:
            self.device = "cpu"
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.enc_lstm = nn.LSTM(enc_input_size, enc_hidden_size, num_layers=num_enc_lstm_layers, dropout=0.3, bidirectional=True,batch_first=True, device=self.device)
        # Size of feature obtained from Word2Vec and embedding layer better be same to contribute equally(i.e given task embedding size and general W2V embedding size is same)
        self.enc_embed_layer = nn.Embedding(vocab_size,embed_size,device=self.device)
        print("self.enc_embed_layer",sum(p.numel() for p in self.enc_embed_layer.parameters()))
        # 8 denotes the number of words u need to contextualize
        # in_channels is the two embeddings obtained. One from embedding layer and another from Word2Vec
        # The kernel size will reduce the width to length 1 and hence out_channels will become new features and Height become new timesteps
        self.enc_cnn = nn.Conv2d(in_channels=2, out_channels=enc_input_size, kernel_size=(8,embed_size),device=self.device)
    
    def forward(self, enc_w2v_embed,enc_ind_embed):
        enc_ind_embed = enc_ind_embed.to(device=self.device, non_blocking=True)
        enc_w2v_embed = enc_w2v_embed.to(device=self.device, non_blocking=True)
        tmp = torch.unsqueeze(enc_ind_embed,dim=2).to(device=self.device, non_blocking=True)
        enc_ind_embed = self.enc_embed_layer(enc_ind_embed)
        # This will make the embedding zero at padding points
        enc_ind_embed = torch.where(tmp == 0,torch.zeros((enc_ind_embed.size()[-1]),device=self.device),enc_ind_embed)
        
        conv_inp = torch.stack((enc_w2v_embed,enc_ind_embed),dim=1)
        # print("conv_inp :{}".format(conv_inp.size()))
        conv_inp = self.enc_cnn(conv_inp)
        # Out channels are the features and height is the number of timesteps
        conv_inp = torch.squeeze(torch.transpose(conv_inp,1,2),-1)
        # print("conv_out :{}".format(conv_inp.size()))
        out,(h_out,c_out) = self.enc_lstm(conv_inp)
        # print("enc_out :{}".format(out.size()))
        return out,(h_out,c_out)


class Dec_LSTM_CNN_Arch(nn.Module):
    def __init__(self, num_dec_lstm_layers,embed_size,dec_hidden_size,vocab_size,is_use_cuda=True):
        super(Dec_LSTM_CNN_Arch, self).__init__()
        if not is_use_cuda:
            self.device = "cpu"
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        self.dec_lstm = nn.LSTM(2*embed_size, dec_hidden_size, num_layers=num_dec_lstm_layers, dropout=0.3, bidirectional=False,batch_first=True, device=self.device)
        self.dec_embed_layer = nn.Embedding(vocab_size,embed_size,device=self.device)
        print("self.dec_embed_layer",sum(p.numel() for p in self.dec_embed_layer.parameters()))
    
    def forward(self, dec_w2v_embed,dec_ind_embed,dec_h_out=None,dec_c_out=None):
        dec_ind_embed = dec_ind_embed.to(device=self.device, non_blocking=True)
        dec_w2v_embed = dec_w2v_embed.to(device=self.device, non_blocking=True)
        dec_ind_embed = self.dec_embed_layer(dec_ind_embed)
        lstm_inp = torch.cat((dec_w2v_embed,dec_ind_embed),dim=2)
        # print("dec_in :{}".format(lstm_inp.size()))
        if(dec_h_out is not None or dec_c_out is not None):
            out,(h_out,c_out) = self.dec_lstm(lstm_inp,(dec_h_out,dec_c_out))
        else:
            out,(h_out,c_out) = self.dec_lstm(lstm_inp)
        # print("dec_out :{}".format(out.size()))
        return out,(h_out,c_out)


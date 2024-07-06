import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import spacy
import string
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import random
import gensim.downloader as api
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ast import literal_eval
import os
from sklearn.feature_extraction.text import CountVectorizer

STRT = "<START>"
END = "<END>"
UNK = "<UNK>"
PAD = ""

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_id + worker_seed)
    random.seed(worker_id - worker_seed)

def get_stopwords():
    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    return stop_words

def mean_wordvector(wv,keys):
    cnt = 0
    mean_wv = None
    for k in keys:
        cv = wv[k]
        if mean_wv is None:
            mean_wv = cv
        else:
            mean_wv = (mean_wv * (cnt-1) + cv)/cnt
        cnt+=1
    
    return mean_wv

def get_extra_token_rep(wv,num_samples):
    random_indices=np.array(random.sample(range(0,len(wv)), num_samples))
    klist = [wv.index_to_key[k] for k in random_indices]
    return mean_wordvector(wv,klist)

def add_UNK_STRT_END_to_W2V(word2vec_obj_list):
    random.seed(10)
    for word_vec_obj in word2vec_obj_list:
        word_vec_obj[UNK] = mean_wordvector(word_vec_obj,word_vec_obj.key_to_index.keys())
        word_vec_obj[STRT] = get_extra_token_rep(word_vec_obj,10)
        word_vec_obj[END] = get_extra_token_rep(word_vec_obj,10)
        word_vec_obj[PAD] = np.zeros_like(word_vec_obj[END])
        # Having the highest frequency count for PAD is necessary in our design bcoz we assume that index 0 indicates PADDING and for that to happen it has to be of highest frequency
        word_vec_obj.set_vecattr(PAD,"count",15000011)
        word_vec_obj.set_vecattr(STRT,"count",15000009)
        word_vec_obj.set_vecattr(END,"count",15000007)
        word_vec_obj.set_vecattr(UNK,"count",15000005)
        
def get_combined_frequency(wordvec_obj_list,key,freq_local_vocab=None):
    overall_freq = 0
    if freq_local_vocab is not None and key not in [STRT,PAD,UNK,END]:
        overall_freq = freq_local_vocab.get(key,0)
    for word_vec_obj in wordvec_obj_list:
        if key in word_vec_obj:
            try:
                overall_freq += word_vec_obj.get_vecattr(key, "count")
            except IndexError:
                print("IndexError",key)
    
    return overall_freq

def form_overall_key_to_index(wordvec_obj_list,freq_local_vocab=None,local_vocab_key_to_indx=None,percentile_to_omit_in_w2v=0):
    overall_keys = set()
    for word_vec_obj in wordvec_obj_list:
        overall_keys.update(list(word_vec_obj.key_to_index.keys()))
    if percentile_to_omit_in_w2v > 0:
        # If percentile to omit is mentioned, remove the bottom x percentile frequency words from global vocab only.
        tmparr = []
        for k in overall_keys:
            tmparr.append(get_combined_frequency(wordvec_obj_list,k,freq_local_vocab))
        qval = np.percentile(np.array(tmparr), percentile_to_omit_in_w2v)
        print("qval",qval)
        subtract_set = set()
        for k in overall_keys:
            cval = get_combined_frequency(wordvec_obj_list,k,freq_local_vocab)
            if(cval<qval):
                subtract_set.add(k)
        overall_keys = overall_keys - subtract_set
        print("len(subtract_set) ",len(subtract_set))
    overall_keys.update(local_vocab_key_to_indx.keys())
    # Order such that highest frequency keys are in the beginning. This is required for doing adaptivelogsoftmax
    ret = sorted(overall_keys,reverse=True, key=lambda item: get_combined_frequency(wordvec_obj_list,item,freq_local_vocab))
    print("ret ",len(ret))
    ret_dict = {ret[i]:i for i in range(len(ret))}
    print("ret_dict ",len(ret_dict))
    return ret_dict,ret

def generate_w2vobjs(list_of_w2v_names):
    word2vec_obj_list = []
    for each_name in list_of_w2v_names:
        cpath = "./data/W2V/"+each_name+".kv"
        if(os.path.exists(cpath)):
            cmodel = gensim.models.KeyedVectors.load(cpath)
        else:
            cmodel = api.load(each_name)
            cmodel.save(cpath)
        print(type(cmodel),cmodel)
        word2vec_obj_list.append(cmodel)
    add_UNK_STRT_END_to_W2V(word2vec_obj_list)
    return word2vec_obj_list

# Skips UNKNOWN words while generating sentence representation
def average_word_rep_sentence_vectorizer(sent_tokens,word_vector_obj_list):
    """
    sent_tokens: 1-D list of tokens each of type str
    word_vector_obj_list : list of word2vec gensim objects
    """
    res_list = []
    for word_vector_obj in word_vector_obj_list:
        vector_size = word_vector_obj.vector_size
        wv_res = np.zeros(vector_size)
        # print(wv_res)
        ctr = 1
        for w in sent_tokens:
            if w in word_vector_obj:
                ctr += 1
                wv_res += word_vector_obj[w]
        wv_res = wv_res/ctr
    res_list = np.concatenate(res_list,axis=-1)
    return res_list

def sentence_vectorizer_using_wordembed(sent_tokens,word_vector_obj_list):
    """
    sent_tokens: 1-D list of tokens each of type str
    word_vector_obj_list : list of word2vec gensim objects
    """
    res_list = []
    for each_token in sent_tokens:
        overall_word_rep = []
        for word_vector_obj in word_vector_obj_list:
            if each_token not in word_vector_obj:
                each_token = UNK
            overall_word_rep.append(word_vector_obj[each_token])

        res_list.append(np.concatenate(overall_word_rep,axis=-1))
    
    return np.stack(res_list)


def generate_local_vocab_word_dict(processed_train_data,list_of_keys_to_merge=["article","highlights"],min_df=0.03):
    # This min_df is important to use. Bcoz otherwise some rare words are part of vocabualary unnecessarily
    vectorizer = CountVectorizer(analyzer=lambda x: x.split(),min_df=min_df)
    df_col_merged = processed_train_data[list_of_keys_to_merge[0]]
    for p in range(1,len(list_of_keys_to_merge)):
        df_col_merged = df_col_merged + processed_train_data[list_of_keys_to_merge[p]]
    corpus_as_list = []
    for index in range(len(df_col_merged)):
        corpus_as_list.append(' '.join(df_col_merged.iloc[index]))
    
    cv_fit = vectorizer.fit_transform(corpus_as_list)
    freq_local_vocab = cv_fit.toarray().sum(axis=0)
    freq_local_vocab = {item:freq_local_vocab[vectorizer.vocabulary_[item]] for item in vectorizer.vocabulary_}

    ret = [PAD,UNK]
    tmp = sorted(vectorizer.vocabulary_.keys(),reverse=True, key=lambda item: freq_local_vocab[item])
    ret.extend(tmp)
    local_vocab_key_to_indx = {ret[i]:i for i in range(len(ret))}

    return local_vocab_key_to_indx,ret,freq_local_vocab

def convert_tokens_to_indices(sent_tokens,overall_key_to_index):
    res_list = []
    for each_token in sent_tokens:
        if each_token not in overall_key_to_index:
                each_token = UNK
        res_list.append(overall_key_to_index[each_token])
    
    return res_list

def fill_in_zero_array(inp,M,N):
    if(M==0):
        return None
    Z = np.zeros((M, N))
    for enu, row in enumerate(inp):
        Z[enu, :len(row)] += row
    return Z


def custom_collate(original_batch):
    
    max_srcvec_seqlen = 0
    max_tarvec_seqlen = 0
    max_srcind_seqlen = 0
    max_tarind_seqlen = 0
    max_labind_seqlen = 0
    for each_batch in original_batch:
        if each_batch[0] is not None:
            max_srcvec_seqlen = max(max_srcvec_seqlen,each_batch[0].shape[0])
        if each_batch[1] is not None:
            max_tarvec_seqlen = max(max_tarvec_seqlen,each_batch[1].shape[0])
        if each_batch[2] is not None:
            max_srcind_seqlen = max(max_srcind_seqlen,len(each_batch[2]))
        if each_batch[3] is not None:
            max_tarind_seqlen = max(max_tarind_seqlen,len(each_batch[3]))
        if each_batch[4] is not None:
            max_labind_seqlen = max(max_labind_seqlen,len(each_batch[4]))
    
    ret_src_vec = []
    ret_tar_vec = []
    ret_src_seq = []
    ret_tar_seq = []
    ret_lab_seq = []
    for each_batch in original_batch:
        src_vec,tar_vec,src_seqind,tar_seqind,lab_seqind = each_batch
        if src_vec is not None:
            ret_src_vec.append(fill_in_zero_array(src_vec,max_srcvec_seqlen,src_vec.shape[1]))
        if tar_vec is not None:
            ret_tar_vec.append(fill_in_zero_array(tar_vec,max_tarvec_seqlen,tar_vec.shape[1]))
        if src_seqind is not None:
            tmp = np.array(src_seqind)
            ret_src_seq.append(np.pad(tmp,(0,max_srcind_seqlen-tmp.shape[0])))
        if tar_seqind is not None:
            tmp = np.array(tar_seqind)
            ret_tar_seq.append(np.pad(tmp,(0,max_tarind_seqlen-tmp.shape[0])))
        if lab_seqind is not None:
            tmp = np.array(lab_seqind)
            ret_lab_seq.append(np.pad(tmp,(0,max_labind_seqlen-tmp.shape[0])))
    
    if len(ret_src_vec)>0:
        ret_src_vec = np.stack(ret_src_vec)
        ret_src_vec = torch.from_numpy(ret_src_vec).float()
    if len(ret_tar_vec)>0:
        ret_tar_vec = np.stack(ret_tar_vec)
        ret_tar_vec = torch.from_numpy(ret_tar_vec).float()
    if len(ret_src_seq)>0:
        ret_src_seq = np.stack(ret_src_seq)
        ret_src_seq = torch.from_numpy(ret_src_seq).long()
    if len(ret_tar_seq)>0:
        ret_tar_seq = np.stack(ret_tar_seq)
        ret_tar_seq = torch.from_numpy(ret_tar_seq).long()
    if len(ret_lab_seq)>0:
        ret_lab_seq = np.stack(ret_lab_seq)
        ret_lab_seq = torch.from_numpy(ret_lab_seq).long()

    return ret_src_vec,ret_tar_vec,ret_src_seq,ret_tar_seq,ret_lab_seq

def convert_seq_indx_to_word(seq_inds,overall_index_to_key):
    ret_seq_arr = []
    for each_seq_ind in seq_inds:
        tmp = []
        for eind in each_seq_ind:
            # If in a sequence u see the <END> tag, truncate the future items
            if(eind == 2):
                break
            else:
                tmp.append(overall_index_to_key[eind])
        if(len(tmp)<5):
            while(len(tmp)<5):
                tmp.append('_')
        ret_seq_arr.append(tmp)
    
    return ret_seq_arr

def convert_seq_arr_to_seq_str(seq_arr):
    ret_strs = []
    for each_sq in seq_arr:
        ret_strs.append(' '.join(each_sq))
    return ret_strs

class TextSummarizationDataset(Dataset):
    def __init__(self, pandas_frame,tokenizer_func,punctuations_to_remove,word_vector_obj_list=[],is_remove_stopwords=False,src_transform=None,target_transform=None,overall_key_to_index=None,local_key_to_index=None,src_sent_key="article",target_sent_key="highlights"):
        """
        pandas_frame: pandas frame to read data from
        src_sent_key: key in pandas frame whose value acts as source sentence
        target_sent_key: key in pandas frame whose value acts as target sentence
        tokenizer_func: Function which tokenizes each key
        punctuations_to_remove: A string which contains characters to remove during tokenization
        word_vector_obj_list: List of word2vec objects which is used during vectorization of tokens
        src_transform and target_transform: Vectorizer on source and target sentences respectively. If None,skip W2v vectorization
        overall_key_to_index: If None,
        """
        self.pandas_frame = pandas_frame
        self.src_key = src_sent_key
        self.target_key = target_sent_key
        self.src_transform = src_transform
        self.target_transform = target_transform
        self.tokenizer_func = tokenizer_func
        self.is_remove_stopwords = is_remove_stopwords
        self.punctuations_to_remove = punctuations_to_remove
        self.word_vector_obj_list = word_vector_obj_list
        self.overall_key_to_index = overall_key_to_index
        self.local_key_to_index = local_key_to_index

    def __len__(self):
        return len(self.pandas_frame)

    def __getitem__(self, idx):
        source_token,target_token = self.pandas_frame.iloc[idx][self.src_key],self.pandas_frame.iloc[idx][self.target_key]
        # This is needed in case the pandas dataframe was converted to string from list during preprocessing save
        if(source_token[0]=='[' and source_token[-1]==']'):
            source_token = literal_eval(source_token)
            target_token = literal_eval(target_token)
        
        if(self.tokenizer_func is not None):
            # Convert sentence/document into a list of token using either lemmatization or stemming
            source_token = self.tokenizer_func(source_token,self.is_remove_stopwords,self.punctuations_to_remove)
            target_token = self.tokenizer_func(target_token,self.is_remove_stopwords,self.punctuations_to_remove)
            # Add STRT and END tag at start and end of sentence/document respectively
            target_token = add_start_end_tags(target_token)
        # Shift label one step to right
        label_token = target_token[1:]
        # Leave out last timestep in target token
        target_token = target_token[:-1]
        # print("{}--{}--{}".format(idx,len(source_token),len(target_token)))
        source_seq = None
        target_seq = None
        source_vec = None
        target_vec = None
        if self.overall_key_to_index is not None and self.local_key_to_index is not None:
            source_seq = convert_tokens_to_indices(source_token,self.local_key_to_index)
            target_seq = convert_tokens_to_indices(target_token,self.local_key_to_index)
            label_seq = convert_tokens_to_indices(label_token,self.overall_key_to_index)
        
        # These transform functions are sentence/doc vectorizers
        if self.src_transform:
            source_vec = self.src_transform(source_token,self.word_vector_obj_list)
        if self.target_transform:
            target_vec = self.target_transform(target_token,self.word_vector_obj_list)
        
        return source_vec, target_vec, source_seq, target_seq,label_seq
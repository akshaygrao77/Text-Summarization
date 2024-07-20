import torch
from collections import OrderedDict

from model_arch import LSTM_CNN_Arch_With_Attention
from utils.data_processing import *
from preprocess_data import spacy_lemmatizer,nltk_stemmer
from vocab_dict_generator import get_vocab_dict,get_vocab_dict_path
import pickle

def perform_inference(model,input_text,wordvec_obj_list,vectorizer_func,index_func,local_vocab_key_to_indx,overall_key_to_index,overall_index_to_key,tokenizer_func,punctuations_to_remove):
    data_token = tokenizer_func(input_text,punctuations_to_remove=punctuations_to_remove)
    print(data_token)
    data_seq = convert_tokens_to_indices(data_token,local_vocab_key_to_indx)
    data_seq = torch.from_numpy(np.array(data_seq)).long()
    data_seq = torch.unsqueeze(data_seq,0)
    # print(data_seq)
    data_vec = vectorizer_func(data_token,wordvec_obj_list)
    data_vec = torch.from_numpy(data_vec).float()
    data_vec = torch.unsqueeze(data_vec,0)
    print(data_vec.shape)
        
    overall_inp = data_vec, None, data_seq, None,None
    outputs_seq_ind,_ = model(overall_inp,wordvec_obj_list,vectorizer_func,index_func,local_vocab_key_to_indx,overall_key_to_index,overall_index_to_key)
    seq_arr = convert_seq_indx_to_word(outputs_seq_ind,overall_index_to_key)
    out_str = convert_seq_arr_to_seq_str(seq_arr)
    print("out_str:{}".format(out_str))

    return out_str

def get_model_from_path(custom_model, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_model = torch.load(model_path, map_location=device)
    if(isinstance(temp_model, torch.nn.DataParallel)):
        new_state_dict = OrderedDict()
        for k, v in temp_model.state_dict().items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        custom_model.load_state_dict(new_state_dict)
    elif(isinstance(temp_model, dict)):
        if("module." in [*temp_model['state_dict'].keys()][0]):
            new_state_dict = OrderedDict()
            for k, v in temp_model['state_dict'].items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            custom_model.load_state_dict(new_state_dict)
        else:
            custom_model.load_state_dict(temp_model['state_dict'])
    else:
        custom_model.load_state_dict(temp_model.state_dict())

    return custom_model


if __name__ == '__main__':
    vectorizer_func = sentence_vectorizer_using_wordembed
    index_func = convert_tokens_to_indices
    tokenizer_func = spacy_lemmatizer
    # tokenizer_func = nltk_stemmer
    is_use_cuda = True
    print(STRT)
    stop_words = get_stopwords()
    punctuations_to_remove = "\"#$%&'()*+-/<=>[\]^_`{|}~"

    w2v_name_list = ['glove-twitter-200','word2vec-google-news-300']
    tokenized_training_dataset_path = "./processed_dataset/spacy_lemmatizer/train_data.pkl"

    vocab_path = get_vocab_dict_path(w2v_name_list,tokenized_training_dataset_path)
    if not os.path.exists(vocab_path):
        vocab_dict = get_vocab_dict(w2v_name_list,tokenized_training_dataset_path)
    else:
        with open(vocab_path, 'rb') as handle:
            vocab_dict = pickle.load(handle)
            if vocab_dict["local_vocab_generated_path"] != tokenized_training_dataset_path or vocab_dict["w2v_name_list"] != w2v_name_list:
                vocab_dict = get_vocab_dict(w2v_name_list,tokenized_training_dataset_path)

    local_vocab_key_to_indx,local_vocab_indx_to_key,freq_local_vocab = vocab_dict["local_vocab_key_to_indx"],vocab_dict["local_vocab_indx_to_key"],vocab_dict["freq_local_vocab"]
    local_vocab_size = len(local_vocab_key_to_indx)
    assert local_vocab_size==len(local_vocab_indx_to_key), "local_vocab_size:{} len(local_vocab_indx_to_key):{}".format(local_vocab_size,len(local_vocab_indx_to_key))
    assert all([local_vocab_key_to_indx[local_vocab_indx_to_key[i]]==i for i in range(local_vocab_size)])
    word2vec_obj_list = vocab_dict["word2vec_obj_list"]
    w2v_vec_size = sum([len(obj[STRT]) for obj in word2vec_obj_list])
    overall_key_to_index,overall_index_to_key = form_overall_key_to_index(word2vec_obj_list,freq_local_vocab,local_vocab_key_to_indx,percentile_to_omit_in_w2v=15)
    vocab_size = len(overall_key_to_index)
    print(local_vocab_key_to_indx[STRT],local_vocab_key_to_indx[UNK],local_vocab_key_to_indx[END],local_vocab_key_to_indx[PAD])
    print(overall_key_to_index[STRT],overall_key_to_index[UNK],overall_key_to_index[END],overall_key_to_index[PAD])
    
    print("##uru",get_combined_frequency(word2vec_obj_list,"##uru",freq_local_vocab=freq_local_vocab))
    print("Ġworld",get_combined_frequency(word2vec_obj_list,"Ġworld",freq_local_vocab=freq_local_vocab))

    model_path = "saved_model/LSTM_CNN_Arch/seq2seq_with_attention.pt"
    model_config = {"num_enc_lstm_layers":3,"embed_size":w2v_vec_size,"enc_input_size":250,"enc_hidden_size":256,"local_vocab_size":local_vocab_size,"vocab_size":vocab_size,"num_dec_lstm_layers":4,"dec_hidden_size":220,"is_use_cuda":is_use_cuda}    

    text_sum_model1 = LSTM_CNN_Arch_With_Attention(model_config)
    text_sum_model1 = get_model_from_path(text_sum_model1,model_path)

    input_text= "The human brain bases confidence, in a given scenario, on doing one’s best to reach a level of achievement that allows the individual to realistically deal with the consequence. The chimp brain, however, believes that an achievement has to be achieved and subsequently it is unable to deal with the consequences."
    perform_inference(text_sum_model1,input_text,word2vec_obj_list,vectorizer_func,index_func,local_vocab_key_to_indx,overall_key_to_index,overall_index_to_key,tokenizer_func,punctuations_to_remove)


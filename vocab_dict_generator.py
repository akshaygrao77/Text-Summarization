from utils.data_processing import *
from preprocess_data import *
import pickle

def get_vocab_dict(w2v_name_list,tokenized_training_dataset_path):
    processed_train_data = pd.read_pickle(tokenized_training_dataset_path)

    local_vocab_key_to_indx,local_vocab_indx_to_key,freq_local_vocab = generate_local_vocab_word_dict(processed_train_data)
    local_vocab_size = len(local_vocab_key_to_indx)
    assert local_vocab_size==len(local_vocab_indx_to_key), "local_vocab_size:{} len(local_vocab_indx_to_key):{}".format(local_vocab_size,len(local_vocab_indx_to_key))
    assert all([local_vocab_key_to_indx[local_vocab_indx_to_key[i]]==i for i in range(local_vocab_size)])
    word2vec_obj_list = generate_w2vobjs(w2v_name_list)
    w2v_vec_size = sum([len(obj[STRT]) for obj in word2vec_obj_list])

    vocab_dict={"local_vocab_key_to_indx":local_vocab_key_to_indx,"local_vocab_indx_to_key":local_vocab_indx_to_key,"freq_local_vocab":freq_local_vocab,
                "word2vec_obj_list":word2vec_obj_list,"local_vocab_generated_path":tokenized_training_dataset_path,"w2v_name_list":w2v_name_list}
    return vocab_dict

def get_vocab_dict_path(w2v_name_list,tokenized_training_dataset_path):
    vocab_path = tokenized_training_dataset_path.replace(".pkl","/")
    tmp = '_'.join(map(str,w2v_name_list)).replace("-","_")
    vocab_path += tmp+"/"
    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)
    vocab_path += "vocab_dict.pkl"
    return vocab_path

if __name__ == '__main__':
    w2v_name_list = ['glove-twitter-200','word2vec-google-news-300']
    tokenized_training_dataset_path = "./processed_dataset/spacy_lemmatizer/train_data.pkl"
    
    vocab_dict = get_vocab_dict(w2v_name_list,tokenized_training_dataset_path)
    vocab_path = get_vocab_dict_path(w2v_name_list,tokenized_training_dataset_path)
    print("vocab_path ",vocab_path)
    with open(vocab_path, 'wb') as handle:
        pickle.dump(vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
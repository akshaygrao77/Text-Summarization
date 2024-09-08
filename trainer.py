import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import wandb
from ignite.metrics import Rouge
import torcheval
from torcheval.metrics.functional.text import bleu
import tqdm
from collections import OrderedDict

from model_arch import LSTM_CNN_Arch_With_Attention,LSTM_CNN_Arch_With_Attention_multiple_span
from utils.data_processing import *
from vocab_dict_generator import *

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def compute_rogue_and_bluescore(overall_index_to_key,rogue_obj,src_seqind,tar_seqind):
    src_seq_arr = convert_seq_indx_to_word(src_seqind,overall_index_to_key)
    tar_seq_arr = convert_seq_indx_to_word(tar_seqind,overall_index_to_key)

    rogue_obj.update((src_seq_arr,[[w] for w in tar_seq_arr]))
    bscore=torcheval.metrics.functional.bleu_score(convert_seq_arr_to_seq_str(src_seq_arr),[[w] for w in convert_seq_arr_to_seq_str(tar_seq_arr)])

    return bscore

def train_model(net, trainloader, validloader,optimizer, epochs, final_model_save_path,overall_index_to_key,local_vocab_key_to_indx,overall_key_to_index, wand_project_name=None,wordvec_obj_list=None,vectorizer_func=None,index_func=None,is_use_cuda=True):
    print("total_params:{} net:{}".format(sum(p.numel() for p in net.parameters()),net))
    if not is_use_cuda:
        device_str = 'cpu'
    else:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cuda':
        print("torch.cuda.device_count() ",torch.cuda.device_count())
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            net = torch.nn.DataParallel(net)

        # !Important: Never switch this ON for multiple span models. Even for plain LSTM based models since model inputs change at each iteration
        # cudnn.benchmark = True
    rogue_obj = Rouge()
    is_log_wandb = not(wand_project_name is None)
    best_rouge_f1score = 0
    net.train()
    # T_max is the number of epochs before restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5*2500)
    for epoch in range(epochs):  # loop over the dataset multiple times
        rogue_obj.reset()
        net.train()

        running_loss = 0.0
        running_bleu_score = 0.0
        loader = tqdm.tqdm(trainloader, desc='Training')
        for batch_idx, data in enumerate(loader, 0):
            begin_time = time.time()
            loader.set_description(f"Epoch {epoch+1}")
            # get the inputs; data is a list of [inputs, labels]
            labels_seqind = data[-1]

            # zero the parameter gradients
            optimizer.zero_grad()
            # pass entire batch of all sequence to model.
            outputs_seq_ind,loss = net(data)
            # print(loss)
            loss=torch.mean(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_bleu_score += compute_rogue_and_bluescore(overall_index_to_key,rogue_obj,outputs_seq_ind,labels_seqind)

            running_loss += loss.item()

            cur_time = time.time()
            step_time = cur_time - begin_time
            loader.set_postfix(train_loss=running_loss/(batch_idx + 1),cur_loss=loss.item(),article_dim=data[0].shape[1],summary_dim=data[1].shape[1],
                               blue_score=running_bleu_score/(batch_idx + 1), stime=format_time(step_time))
            # if(batch_idx>1):
            #     break
        
        train_loss = running_loss/(batch_idx + 1)
        train_bleu_score = running_bleu_score/(batch_idx + 1)
        train_roug_scores = rogue_obj.compute()

        print("train_loss:{} train_bleu_score:{} train_roug_scores:{} ".format(train_loss,train_bleu_score,train_roug_scores))
        per_epoch_model_save_path = final_model_save_path.replace(
            ".pt", "")
        if not os.path.exists(per_epoch_model_save_path):
            os.makedirs(per_epoch_model_save_path)
        per_epoch_model_save_path += "/epoch_{}_dir.pt".format(epoch)
        if(epoch % 1 == 0):
            torch.save(net, per_epoch_model_save_path)
        test_bleu_score, test_roug_scores = evaluate_model(
            net, validloader,local_vocab_key_to_indx,overall_key_to_index,overall_index_to_key,wordvec_obj_list,vectorizer_func,index_func)
        print(" valid_bleu_score:{} valid_roug_scores:{}".format(test_bleu_score,test_roug_scores))
        if(is_log_wandb):
            wandb.log({"cur_epoch":epoch+1,"train_loss":train_loss,"train_bleu_score": train_bleu_score, "valid_bleu_score": test_bleu_score,"train_roug_scores":train_roug_scores,"valid_roug_scores":test_roug_scores})

        if(test_roug_scores["Rouge-L-F"] >= best_rouge_f1score):
            best_rouge_f1score = test_roug_scores["Rouge-L-F"]
            torch.save(net, final_model_save_path)

    print('Finished Training: Best saved model test best_rouge_f1score is:', best_rouge_f1score)
    return best_rouge_f1score, torch.load(final_model_save_path)

def evaluate_model(net, dataloader,local_vocab_key_to_indx,overall_key_to_index,overall_index_to_key,wordvec_obj_list=None,vectorizer_func=None,index_func=None):
    net.eval()
    
    bleu_score = 0
    rogue_obj = Rouge()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        loader = tqdm.tqdm(dataloader, desc='Testing')
        for batch_idx, data in enumerate(loader, 0):
            begin_time = time.time()
            labels_seqind = data[-1]

            data = data[0],None,data[2],None,None
            # calculate outputs by running images through the network
            outputs_seq_ind,_ = net(data,wordvec_obj_list,vectorizer_func,index_func,local_vocab_key_to_indx,overall_key_to_index,overall_index_to_key)

            bleu_score += compute_rogue_and_bluescore(overall_index_to_key,rogue_obj,outputs_seq_ind,labels_seqind)
            cur_time = time.time()
            step_time = cur_time - begin_time
            loader.set_postfix(overall_dim=data[0].shape[1],
                               blue_score=bleu_score/(batch_idx + 1), stime=format_time(step_time))
            # if(batch_idx>1):
            #     break
            

    return bleu_score/(batch_idx + 1),rogue_obj.compute()

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
    wand_project_name = "Text_Summarization"
    # wand_project_name = None
    w2v_name_list = ['glove-twitter-200','word2vec-google-news-300']
    vectorizer_func = sentence_vectorizer_using_wordembed
    index_func = convert_tokens_to_indices

    start_net_path = None
    # start_net_path = "saved_model/LSTM_CNN_Arch/seq2seq_with_attention.pt"
    
    batch_size = 128
    epochs = 50
    is_use_cuda = True
    print(STRT)
    stop_words = get_stopwords()
    punctuations_to_remove = "\"#$%&'()*+-/<=>[\]^_`{|}~"

    tokenized_training_dataset_path = './processed_dataset/spacy_lemmatizer/train_data.pkl'
    processed_train_data = pd.read_pickle(tokenized_training_dataset_path)
    processed_valid_data = pd.read_pickle('./processed_dataset/spacy_lemmatizer/validation.pkl')
    processed_test_data = pd.read_pickle('./processed_dataset/spacy_lemmatizer/test.pkl')

    vocab_path = get_vocab_dict_path(w2v_name_list,tokenized_training_dataset_path)
    if not os.path.exists(vocab_path):
        vocab_dict = get_vocab_dict(w2v_name_list,tokenized_training_dataset_path)
        with open(vocab_path, 'wb') as handle:
            pickle.dump(vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(vocab_path, 'rb') as handle:
            vocab_dict = pickle.load(handle)
            if vocab_dict["local_vocab_generated_path"] != tokenized_training_dataset_path or vocab_dict["w2v_name_list"] != w2v_name_list:
                vocab_dict = get_vocab_dict(w2v_name_list,tokenized_training_dataset_path)
                with open(vocab_path, 'wb') as handle:
                    pickle.dump(vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    local_vocab_key_to_indx,local_vocab_indx_to_key,freq_local_vocab = vocab_dict["local_vocab_key_to_indx"],vocab_dict["local_vocab_indx_to_key"],vocab_dict["freq_local_vocab"]
    local_vocab_size = len(local_vocab_key_to_indx)
    assert local_vocab_size==len(local_vocab_indx_to_key), "local_vocab_size:{} len(local_vocab_indx_to_key):{}".format(local_vocab_size,len(local_vocab_indx_to_key))
    assert all([local_vocab_key_to_indx[local_vocab_indx_to_key[i]]==i for i in range(local_vocab_size)])
    word2vec_obj_list = vocab_dict["word2vec_obj_list"]
    w2v_vec_size = sum([len(obj[STRT]) for obj in word2vec_obj_list])
    overall_key_to_index,overall_index_to_key = form_overall_key_to_index(word2vec_obj_list,freq_local_vocab,local_vocab_key_to_indx,percentile_to_omit_in_w2v=15)
    vocab_size = len(overall_key_to_index)
    print("overall_index_to_key ",overall_index_to_key[:100])

    train_ts_spacy_ds = TextSummarizationDataset(processed_train_data,None,punctuations_to_remove,word2vec_obj_list,src_transform=vectorizer_func,
        target_transform=vectorizer_func,overall_key_to_index=overall_key_to_index,local_key_to_index = local_vocab_key_to_indx)
    train_seed_gen = torch.Generator()
    train_seed_gen.manual_seed(2022)
    train_dataloader = torch.utils.data.DataLoader(
            train_ts_spacy_ds, shuffle=True, pin_memory=True, num_workers=16, batch_size=batch_size,collate_fn=custom_collate,generator=train_seed_gen, worker_init_fn=seed_worker)
    
    valid_ts_spacy_ds = TextSummarizationDataset(processed_valid_data,None,punctuations_to_remove,word2vec_obj_list,src_transform=vectorizer_func,
        target_transform=vectorizer_func,overall_key_to_index=overall_key_to_index,local_key_to_index = local_vocab_key_to_indx)
    valid_dataloader = torch.utils.data.DataLoader(
            valid_ts_spacy_ds, shuffle=False, pin_memory=False, num_workers=4, batch_size=batch_size,collate_fn=custom_collate)
    
    print("local_vocab_size:{} vocab_size:{}".format(local_vocab_size,vocab_size))
    model_config = {"no_of_encs":4,"num_enc_lstm_layers":3,"embed_size":w2v_vec_size,"enc_input_size":250,"enc_hidden_size":256,"local_vocab_size":local_vocab_size,"vocab_size":vocab_size,"num_dec_lstm_layers":4,"dec_hidden_size":220,"is_use_cuda":is_use_cuda}
    
    if(model_config["enc_hidden_size"] % model_config["no_of_encs"] != 0):
        raise Exception("Invalid no_of_encs:{} or enc_hidden_size:{}".format(no_of_encs,enc_hidden_size))
    if(start_net_path is not None):
        text_sum_model1 = LSTM_CNN_Arch_With_Attention_multiple_span(model_config)
        # text_sum_model1 = LSTM_CNN_Arch_With_Attention(model_config)
        text_sum_model1 = get_model_from_path(text_sum_model1,start_net_path)
    else:
        text_sum_model1 = LSTM_CNN_Arch_With_Attention_multiple_span(model_config)
        # text_sum_model1 = LSTM_CNN_Arch_With_Attention(model_config)

    optimizer = optim.Adam(text_sum_model1.parameters(), lr=0.01)
    final_model_save_path = "./saved_model/LSTM_CNN_Arch_multiple_span/seq2seq_with_attention.pt"

    is_log_wandb = not(wand_project_name is None)
    if(is_log_wandb):
        wandb_config = dict()
        wandb_config["optimizer"] = optimizer
        wandb_config["final_model_save_path"] = final_model_save_path
        wandb_config["epochs"] = epochs
        wandb_config["batch_size"] = batch_size
        wandb_config["arch"] = str(text_sum_model1)
        wandb_config["arch_type"] = str(type(text_sum_model1))
        wandb_config["w2v_name_list"] = str(w2v_name_list)
        wandb_config["vectorizer_func"] = vectorizer_func
        wandb_config["index_func"] = index_func
        wandb_config["start_net_path"] = start_net_path
        wandb_config["data_source"]="./processed_dataset/spacy_lemmatizer/train_data.csv"
        wandb_config.update(model_config)

        wandb.init(
            project=f"{wand_project_name}",
            config=wandb_config,
        )

    rscore,net = train_model(text_sum_model1, train_dataloader, valid_dataloader,optimizer, epochs, final_model_save_path,overall_index_to_key,local_vocab_key_to_indx,overall_key_to_index,wand_project_name,word2vec_obj_list,vectorizer_func,index_func,is_use_cuda)

    test_ts_spacy_ds = TextSummarizationDataset(processed_valid_data,None,punctuations_to_remove,word2vec_obj_list,src_transform=vectorizer_func,
        target_transform=vectorizer_func,overall_key_to_index=overall_key_to_index,local_key_to_index = local_vocab_key_to_indx)
    test_dataloader = torch.utils.data.DataLoader(
            test_ts_spacy_ds, shuffle=False, pin_memory=True, num_workers=4, batch_size=batch_size,collate_fn=custom_collate)

    test_bleu_score, test_roug_scores = evaluate_model(text_sum_model1, test_dataloader,local_vocab_key_to_indx,overall_key_to_index,overall_index_to_key,word2vec_obj_list,vectorizer_func,index_func)
    if(is_log_wandb):
        wandb.log({"best_valid_roug_scores": rscore,"test_bleu_score":test_bleu_score,"test_roug_scores":test_roug_scores})
        wandb.finish()
    
    print("Execution completed!")
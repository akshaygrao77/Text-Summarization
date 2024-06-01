import pandas as pd
import gensim
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import spacy
import string
import nltk
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import random
import gensim.downloader as api
from pandarallel import pandarallel

def spacy_lemmatizer(sentence,is_remove_stopword=False,punctuations_to_remove="",verbose=False):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]
    if(verbose):
        print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if (not is_remove_stopword or word not in stop_words) and word not in punctuations_to_remove ]

    # return preprocessed list of tokens
    return mytokens

def nltk_stemmer(sentence,is_remove_stopword=False,punctuations_to_remove="",verbose=False):
    stemmer_obj = SnowballStemmer(language='english')
    mytokens = word_tokenize(sentence)
    # print(mytokens)
    # Removing stop words and punctuations
    mytokens = [ stemmer_obj.stem(word) for word in mytokens if (not is_remove_stopword or word not in stop_words) and word not in punctuations_to_remove ]

    # return preprocessed list of tokens
    return mytokens

def add_start_end_tags(sent_tokens):
    sent_tokens.insert(0,STRT)
    sent_tokens.append(END)
    return sent_tokens


def generate_tokenized_dataset(pdframe,save_path,tokenizer_func,punctuations_to_remove=""):
    pandarallel.initialize(progress_bar=True)
    print("Processing article tag, will be saved at :{}".format(save_path))
    # Replace article with the list of tokenizer version
    pdframe["article"] = pdframe["article"].parallel_apply(lambda x: tokenizer_func(x,punctuations_to_remove=punctuations_to_remove))
    print("Processing highlights tag, will be saved at :{}".format(save_path))
    pdframe["highlights"] = pdframe["highlights"].parallel_apply(lambda x: add_start_end_tags(tokenizer_func(x,punctuations_to_remove=punctuations_to_remove)))

    filepath = Path(save_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pdframe.to_csv(filepath)
    save_path = save_path.replace(".csv",".pkl")
    filepath = Path(save_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pdframe.to_pickle(filepath)

if __name__ == '__main__':
    STRT = "<START>"
    END = "<END>"
    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    punctuations_to_remove = "\"#$%&'()*+-/<=>[\]^_`{|}~"

    train_data = pd.read_csv('./data/CNN-DailyMail News Dataset/train.csv')
    test_data = pd.read_csv('./data/CNN-DailyMail News Dataset/test.csv')
    valid_data = pd.read_csv('./data/CNN-DailyMail News Dataset/validation.csv')

    # Change the tokenizer type to generate different processed_dataset
    generate_tokenized_dataset(train_data,"./processed_dataset/spacy_lemmatizer/train_data.csv",spacy_lemmatizer,punctuations_to_remove)
    generate_tokenized_dataset(valid_data,"./processed_dataset/spacy_lemmatizer/validation.csv",spacy_lemmatizer,punctuations_to_remove)
    generate_tokenized_dataset(test_data,"./processed_dataset/spacy_lemmatizer/test.csv",spacy_lemmatizer,punctuations_to_remove)
    print("Execution completed!")
    

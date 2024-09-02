# Text-Summarization
## Some of the design features and enhancements incorporated are:
1. Used CNNs to encode the representation of a span of tokens in the encoder. Each 2D filter in CNN acts on passage embedding to obtain a scalar representation of the current span. So, K CNN filters produce K distinct features at each span, which are later used as input to Bi-Directional LSTM.
2. Constructing vocabulary only on the relatively smaller training dataset might perform poorly when out-of-vocabulary words are encountered in real time. To solve this, I used Gensim libraries Word2Vec’s. This doesn’t just increase the vocabulary multi-fold (called global vocabulary) and provides token representations obtained using word2vec methods on massive datasets. I used a combination of glove-twitter-200 and google-news-300 embeddings. So, the overall word2vec representation was a concatenation of these two embeddings.
3. Due to using a large global vocabulary size (3495379) at the prediction layer, normal softmax is too expensive. So, I used the AdaptiveLogSoftmax module from Pytorch, which performs hierarchical softmax operations to allocate more computing model capacity to the more frequent words.
4. Used learnt embedding over local vocabulary to learn task-specific input embeddings and fused it with a more generic word2vec representation for enhanced representational power.


## Some minor details and caveats handled:
1. Add STRT, UNK, END, and PAD tokens to local and global vocabulary. Assign zero vector representation for PAD tokens in both local and global vocabulary. This avoids filling zero at padding indices during multi-head attention operations. UNK token representation is the mean vector of all word2vec representations. The representation of the STRT and END tokens is obtained as the mean vector representation of 10 random keys.
2. Using AdaptiveLogSoftmax requires indexing global vocabulary such that the most frequent tokens are indexed from 0.
3. Reduce vocabulary by removing tokens that are at the lower 15 percentile of frequencies
4. One needs key_to_index and index_to_key mappings overall. Form this considering local and global vocabularies together(perform indexing considering the joint frequencies of local and global word occurrences).
5. Have a specific utility for lemmatizing and storing processed datasets since it's an expensive operation, unlike stemming.
6. Have a specific utility for generating vocabulary and storing processed vocabulary since it's again an expensive operation, and at inference, this should be readily available.
7. It is important not to consider loss at padded indices. Also, once the END token is seen in the generated sequence during training/inference, don't include further tokens in the metric calculations
8. Adding the STRT, END, UNK and PAD tokens is better handled at each batch rather than the entire dataset because the number of padded tokens reduces if done per batch. This also allows for the shift of inputs for the encoder and decoder on the fly.
9. AdaptiveLogSoftmax doesn't handle batched inputs. So don't iterate and merge results one batch at a time. This is because it is hard to set a batch size this way to remain inside memory constraints. This is because the number of timesteps might change rapidly from one batch to another, changing memory footprints heavily during training inside adaptive softmax. So, in the adaptive softmax layer, iterate one timestep at a time. With this modification, the memory footprint stays the same inside adaptive softmax from one batch to another.
10. Local key_to_index and index_to_key mappings are used to learn input embeddings at encoder and decoder modules. This is necessary since global key_to_indx is large and learning that at input embeddings would be too expensive.

## Results
ROUGE-L-F score on (validation set,training set) was (0.25,0.38) and BLEU score on (validation set, training set) was (9.5,15.8)

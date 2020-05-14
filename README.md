# quora-question-duplicates
Evaluating different NLP methods on Quora's question duplicate task

| Model type                                                           | Test accuracy | Improvement over random |
|----------------------------------------------------------------------|---------------| ----------------------- |
| True random                                                          | 50%           | 0%
| Logistic regression (features: cosine similarity, length difference) | 65.54%        | 31.08% |
| Cosine similarity from embeddings (not pre-trained) | 54.53% | 9.06% |
| Cosine similarity from GloVe embeddings<sup>1<sup> | 50.22% | 0.44%
| RNN with tanh activation<sup>2<sup> | 63.08% | 26.16%
| RNN with tanh activation (+ bidirectional) | 73.34% | 46.68% |
| GRU | 77.18% | 54.36% |
| GRU (+ bidirectional) | 77.37% | 54.74% |
| LSTM | 76.52% | 53.04% |
| LSTM (+ bidirectional) | 76.96% | 53.92% |
| CNN | 79.83% | 59.66% |
| BERT base uncased (pre-trained)<sup>1, 3<sup> | 67.12% | 34.24% |


1: model checkpoint is too big to push, so none is provided

2: PyTorch also has RNNs with ReLU activation, which I attempted to implement. Though everything complies, the loss
is huge and the model fails to learn, probably due to some problem with an exploding or vanishing gradient

3: I assume there's something wrong with the BERT implementation because generally we'd expect using pre-trained BERT
weights to give us better results than our own trained models. However, I tried lots of different things, and my code
seems to follow what others use, and still got around the same accuracy.

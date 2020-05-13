# quora-question-duplicates
Evaluating different NLP methods on Quora's question duplicate task

| Model type                                                           | Test accuracy |
|----------------------------------------------------------------------|---------------|
| True random                                                          | 50%           |
| Logistic regression (features: cosine similarity, length difference) | 65.54%        |
| Cosine similarity from embeddings (not pre-trained) | 54.53% |
| Cosine similarity from GloVe embeddings\* | 50.22% |
| RNN with tanh activation\** | 63.08% |
| RNN with tanh activation (+ bidirectional) | 73.34% |
| GRU | 77.18% |
| GRU (+ bidirectional) | 77.37% |
| LSTM | 76.52% |
| LSTM (+ bidirectional) | 76.96% |
| CNN | 79.83% |


\* model checkpoint is too big to push, so none is provided

\** PyTorch also has RNNs with ReLU activation, which I attempted to implement. Though everything complies, the loss
is huge and the model fails to learn, probably due to some problem with an exploding or vanishing gradient

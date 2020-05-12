# quora-question-duplicates
Evaluating different NLP methods on Quora's question duplicate task

| Model type                                                           | Test accuracy |
|----------------------------------------------------------------------|---------------|
| True random                                                          | 50%           |
| Logistic regression (features: cosine similarity, length difference) | 65.54%        |
| Cosine similarity from embeddings (not pre-trained) | 54.53% |
| Cosin similarity from GloVe embeddings\* | 50.22% |

\* model checkpoint is too big to push, so none is provided

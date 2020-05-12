import torch
import torch.nn as nn


class EmbedCosSim(nn.Module):
    def __init__(self, text_field, embedding_dim):
        super(EmbedCosSim, self).__init__()
        self.embedding_dim = embedding_dim
        self.text_field = text_field
        self.vocab_size = len(text_field.vocab)

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.out = nn.Linear(embedding_dim, 1)
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, batch):
        embedded_q1, embedded_q2 = self.embedding(batch.question1), self.embedding(batch.question2)
        out_q1, out_q2 = self.out(embedded_q1), self.out(embedded_q2)

        return torch.sigmoid(self.cos(out_q1, out_q2).squeeze(1))

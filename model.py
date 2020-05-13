import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe


class EmbedCosSim(nn.Module):
    def __init__(self, text_field, embedding_dim, use_glove, glove_dim, checkpoint_name):
        super(EmbedCosSim, self).__init__()
        self.checkpoint_name = checkpoint_name
        self.text_field = text_field
        self.vocab_size = len(text_field.vocab)
        self.embedding_dim = embedding_dim
        if use_glove:
            glove_emb = GloVe('6B', dim=glove_dim)
            self.embedding = nn.Embedding.from_pretrained(glove_emb.vectors, freeze=True)
            self.out = nn.Linear(glove_dim, 1)
        else:
            self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
            self.out = nn.Linear(embedding_dim, 1)
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, batch):
        embedded_q1, embedded_q2 = self.embedding(batch.question1), self.embedding(batch.question2)
        out_q1, out_q2 = self.out(embedded_q1), self.out(embedded_q2)

        return torch.sigmoid(self.cos(out_q1, out_q2).squeeze(1))


class RNNClassifier(nn.Module):
    def __init__(self, text_field, embedding_dim, hidden_dim, rnn_type, bidir, checkpoint_name):
        super(RNNClassifier, self).__init__()
        self.checkpoint_name = checkpoint_name
        self.text_field = text_field
        self.vocab_size = len(text_field.vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidir = bidir
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        if rnn_type == "RNN_TANH":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, nonlinearity='tanh', bidirectional=bidir)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, bidirectional=bidir)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidir)
        else:
            raise ValueError("Please choose either RNN_TANH, GRU, or LSTM as RNN type")

        if bidir:
            self.out = nn.Linear(hidden_dim * 2, 1)
        else:
            self.out = nn.Linear(hidden_dim, 1)

    def forward(self, batch):
        question = torch.cat((batch.question1, batch.question2), dim=0)
        embedded = self.embedding(question)
        if self.rnn_type == "LSTM":
            rnn_out, (hidden, cell) = self.rnn(embedded)
        else:
            rnn_out, hidden = self.rnn(embedded)

        if self.bidir:
            hidden_final = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            output = self.out(hidden_final)
        else:
            hidden_final = hidden[-1, :, :]
            output = self.out(hidden_final)

        return torch.sigmoid(output.squeeze(1))


class CNNClassifier(nn.Module):
    def __init__(self, text_field, embedding_dim, num_filters, filter_sizes, checkpoint_name):
        super(CNNClassifier, self).__init__()
        self.checkpoint_name = checkpoint_name
        self.text_field = text_field
        self.vocab_size = len(text_field.vocab)
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes

        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.convolutions = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (K, embedding_dim), padding=(K - 1, 0)) for K in filter_sizes]
        )
        self.out = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self, batch):
        question = torch.cat((batch.question1, batch.question2), dim=0)
        embedded = self.embedding(question)
        embedded = embedded.permute(1, 0, 2).unsqueeze(1)

        conv_layer = [F.relu(conv(embedded)).squeeze(3) for conv in self.convolutions]
        conv_pool = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conv_layer]
        conv_out = torch.cat(conv_pool, 1)

        output = self.out(conv_out)
        return torch.sigmoid(output.squeeze(1))

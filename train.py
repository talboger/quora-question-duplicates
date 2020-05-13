import torch
import torch.nn as nn
import torch.optim as optim
import time
from tools import prep_torch_data
from model import EmbedCosSim, RNNClassifier, CNNClassifier
from early_stopping import EarlyStopping


def epoch_time(start_time, end_time):
    # gets time an epoch ran for
    elapsed = end_time - start_time
    elapsed_min = int(elapsed / 60)
    elapsed_sec = int(elapsed - (elapsed_min * 60))
    return elapsed_min, elapsed_sec


def train_epoch(model, iterator, optimizer, criterion, short_train):
    # trains one epoch of the model
    model.train()
    epoch_loss = 0

    for n, batch in enumerate(iterator):
        if short_train and n % 50 != 0:
            continue
        batch_loss = torch.tensor(0., requires_grad=True)
        optimizer.zero_grad()
        output = model(batch)
        batch_loss = batch_loss + criterion(output, batch.is_duplicate.float())
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    # evaluates model on valid and test sets
    model.eval()
    epoch_loss = 0
    num_correct, total_pred = 0, 0
    with torch.no_grad():
        for batch in iterator:
            output = model(batch)
            epoch_loss = epoch_loss + criterion(output, batch.is_duplicate.float()).item()
            pred_labels = (output > 0.5).type(torch.int)
            num_correct = num_correct + torch.sum(pred_labels == batch.is_duplicate).item()
            total_pred = total_pred + len(output)
    return epoch_loss / len(iterator), (num_correct / total_pred) * 100


def train(model, train_iter, val_iter, test_iter, optimizer, criterion, n_epochs, short_train,
          checkpoint_name, patience):
    early_stopping = EarlyStopping(filename=checkpoint_name, patience=patience)
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_iter, optimizer, criterion, short_train)
        val_loss, val_acc = evaluate(model, val_iter, criterion)
        end_time = time.time()

        epoch_min, epoch_sec = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_min}m {epoch_sec}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f} | Val. Accuracy {val_acc:.3f}')

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping, reloading checkpoint model")
            model.load_state_dict(torch.load(checkpoint_name))
            break

    test_loss, test_acc = evaluate(model, test_iter, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc:.3f}')


if __name__ == "__main__":
    # load data and define model
    train_iter, val_iter, test_iter, text_field, label_field = prep_torch_data(batch_size=32)
    embedding_dim = 64
    hidden_dim = 32

    criterion = nn.BCELoss()
    # model = EmbedCosSim(text_field, embedding_dim, use_glove=False, glove_dim=100,
    #                     checkpoint_name="checkpoints/embed_cos_sim.pt")  # for training model without GloVe
    # model = EmbedCosSim(text_field, embedding_dim, use_glove=True, glove_dim=100,
    #                   checkpoint_name='checkpoints/embed_cos_sim_glove.pt')  # for training model with GloVe
    # model = RNNClassifier(text_field, embedding_dim, hidden_dim, rnn_type="GRU", bidir=False,
    #                      checkpoint_name='checkpoints/gru.pt')
    # in the above line, you can change rnn_type to either RNN_TANH, GRU, or LSTM to create a different network
    # you can also set bidir=True to create a bidirectional network

    model = CNNClassifier(text_field, embedding_dim, num_filters=32, filter_sizes=[1, 2, 3, 5],
                          checkpoint_name='checkpoints/cnn.pt')

    optimizer = optim.Adam(model.parameters())
    # move everything to gpu if available
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        model.cuda()
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train(model, train_iter, val_iter, test_iter, optimizer, criterion, n_epochs=50, short_train=True,
          checkpoint_name=model.checkpoint_name, patience=5)

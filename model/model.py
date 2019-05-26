from torch.utils.data import Dataset
import torch.nn as nn
import torch
import pickle


class Voc(Dataset):
    """A trivial dataset to give the pickle file something to read into"""
    def __init__(self):
        self.trivial = None


def han_prediction(sentence, model_path, vocab_path):
    """Given a sentence for prediction, the path to the model
    embeddings and the path to the vocab dictionary, this function
    will return the probability of positivity and the word importances"""

    class HAN(nn.Module):
        def __init__(self, vocab_size, embedding_dim=60,
                     hidden_size=15, num_layers=1, dropout=0.1):
            super(HAN, self).__init__()
            self.gru = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=True,
                              dropout=(0 if num_layers == 1 else dropout),
                              bidirectional=True)

            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.fc1 = nn.Linear(2 * hidden_size, 100)
            self.tanh = nn.Tanh()
            self.fc2 = nn.Linear(100, 1, bias=False)
            self.softmax = nn.Softmax(dim=1)
            self.fc3 = nn.Linear(2 * hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, input_seq, input_lengths, hidden=None):
            # convert word indices to embeddings
            embedded = self.embedding(input_seq)

            # pass embeddings through GRU
            # all_layers is shape
            # batch_size x max(sentence_length) x hidden_size*2
            all_layers, final = self.gru(embedded, hidden)

            # pass the hidden layers through linear layer
            # each word of each document will now be a length 100 tensor
            u = self.tanh(self.fc1(all_layers))

            # map the length 100 tensor to a scalar representing importance
            # take the softmax to get the word importance WRT the document
            alpha = self.softmax(self.fc2(u))

            # take sum of hidden layers each sentence weighted by the alphas
            s = (all_layers * alpha).sum(dim=1)

            # take the linear combination of hidden layers and
            # plug into Linear Layer and Sigmoid to get probability
            # s = self.sigmoid(self.fc3(s))
            s = self.fc3(s)

            return s.squeeze(), alpha.squeeze()

    def han_(sentence):
        # slightly hacky way to use the word "a" as a default
        sentence_tensor = torch.Tensor([reviews.word2index.get(word, 5)
                                        for word in sentence.split()]).long()
        sentence_tensor = sentence_tensor.unsqueeze(dim=0)

        p, alpha = han(sentence_tensor, [len(sentence.split())])
        return p.tolist(), alpha.tolist()

    reviews = pickle.load(open(vocab_path, 'rb'))
    V = len(reviews.index2word)

    han = HAN(hidden_size=40, vocab_size=V, embedding_dim=50)
    han.load_state_dict(torch.load(model_path))
    han.eval()

    return han_(sentence)


if __name__ == '__main__':
    sample = 'wonderful movie for the whole family'
    p_, alpha_ = han_prediction(sample, './HAN_lower.pt', './reviews.pkl')

    print(p_)
    print(alpha_)

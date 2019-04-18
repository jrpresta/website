from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pickle


MAX_LENGTH = 15

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3


class Voc(Dataset):
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS',
                           EOS_token: 'EOS', UNK_token: 'UNK'}
        self.num_words = 4  # include the ones above

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word.lower())

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # remove words that appear less frequently
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS',
                           EOS_token: 'EOS', UNK_token: 'UNK'}
        self.num_words = 4
        for word in keep_words:
            self.add_word(word)

def han_prediction(sentence, model_path, vocab_path):

    class HAN(nn.Module):
        def __init__(self, vocab_size, embedding_dim=60,
                     hidden_size=15, num_layers=1, dropout=0.1):
            super(HAN, self).__init__()
            self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,
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
            # all_layers is shape batch_size x max(sentence_length) x hidden_size*2
            all_layers, final = self.gru(embedded, hidden)

            # pass the hidden layers through linear layer
            # each word of each document will now be a length 100 tensor
            u = self.tanh(self.fc1(all_layers))

            # map the length 100 tensor to a scalar representing importance
            # take the softmax to get the word importance WRT the document
            alpha = self.softmax(self.fc2(u))

            # take sum of hidden layers for each sentence weighted by the alphas
            s = (all_layers * alpha).sum(dim=1)

            # take the linear combination of hidden layers and
            # plug into Linear Layer and Sigmoid to get probability
            s = self.sigmoid(self.fc3(s))

            return s.squeeze(), alpha.squeeze()


    # takes string sentence and returns sentence of word indices
    def indexesFromSentence(voc, sentence):
        return [voc.word2index[word] for word in sentence.split()] + [EOS_token]


    def han_(sentence):
        sentence_tensor = torch.Tensor([reviews.word2index[word] for word in sentence.split()]).long()
        sentence_tensor = sentence_tensor.unsqueeze(dim=0)

        p, alpha = han(sentence_tensor, [len(sentence.split())])
        return p.tolist(), alpha.tolist()


    reviews = pickle.load(open(vocab_path, 'rb'))
    print(reviews)
    V = len(reviews.index2word)

    han = HAN(hidden_size=40, vocab_size=V, embedding_dim=50)
    han.load_state_dict(torch.load(model_path))
    han.eval()

    return han_(sentence)


if __name__ == '__main__':
    sample = 'wonderful movie for the whole family'
    p, alpha = han_prediction(sample, './HAN_lower.pt', './reviews.pkl')

    print(p)
    print(alpha)
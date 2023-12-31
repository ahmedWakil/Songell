import torch
import torch.nn as nn
import random
from vcodec import Vcodec


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k_catagories):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # my model structure
        self.rnn_layer = nn.RNN(k_catagories+input_size,
                                hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        # forward pass through the model
        input_combined = torch.cat((category, input), 1)
        output, hidden = self.rnn_layer(input_combined, hidden)
        output = self.fc(output)
        output = self.dropout(output)
        output = self.softmax(output)

        return output, hidden

    def trainWord(self, criterion, optimizer, category_tensor, input_tensor, target_tensor):
        target_tensor.unsqueeze_(-1)
        word_length = input_tensor.size(0)
        hidden = self.initHidden()

        loss = torch.Tensor([0])

        for i in range(word_length):
            # get each prediction and add to the loss at each time step
            output, hidden = self(category_tensor, input_tensor[i], hidden)
            L = criterion(output, target_tensor[i])
            loss += L

        # back propagation
        loss.backward()
        # opdate perameters
        optimizer.step()
        self.zero_grad()

        return output, loss.item()/input_tensor.size(0)

    def sampleWord(self, category, codec: Vcodec):

        max_length = 40
        output_name = ''
        input_tensor = codec.encodeWord(codec.sos)
        category_tensor = codec.catagoryTensor(category)
        hn = self.initHidden()

        for i in range(max_length):
            # print(f'Iteration number: {i}')
            out, hn = self(category_tensor, input_tensor[0], hn)

            p = torch.exp(out)

            topv, topi = out.topk(1)
            topi = topi[0][0]
            # print(f'topv: {topv}\ntopi: {topi}')

            letter = codec.char_set[topi]
            # print(f'prediction: {letter}')
            if letter == codec.eos:
                break

            input_tensor = codec.encodeWord(letter)
            output_name += letter

        return output_name

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k_catagories):
        super(LSTMModel, self).__init__()
        self.hiddensize = hidden_size
        self.context = nn.Linear(k_catagories+input_size, input_size)
        self.lstm_layer = nn.LSTM(input_size, hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden, cell):
        input_combined = torch.cat((category, input), 1)
        out = self.context(input_combined)
        out, (hn, cn) = self.lstm_layer(out, (hidden, cell))
        out = self.fc(out)
        out = self.dropout(out)
        out = self.softmax(out)

        return out, hn, cn

    def trainWord(self, criterion, optimizer, category_tensor, input_tensor, target_tensor):
        target_tensor.unsqueeze_(-1)
        word_length = input_tensor.size(0)
        hidden = self.initHidden()
        cell = self.initHidden()

        loss = torch.Tensor([0])

        for i in range(word_length):
            # get each prediction and add to the loss at each time step
            output, hidden, cell = self(
                category_tensor, input_tensor[i], hidden, cell)
            L = criterion(output, target_tensor[i])
            loss += L

        # back propagation
        loss.backward()
        # opdate perameters
        optimizer.step()
        self.zero_grad()

        return output, loss.item()/input_tensor.size(0)

    def sampleWord(self, category, codec: Vcodec, max_length=40):

        output_name = ''
        input_tensor = codec.encodeWord(codec.sos)
        category_tensor = codec.catagoryTensor(category)
        hn = self.initHidden()
        cn = self.initHidden()

        for i in range(max_length):
            out, hn, cn = self(category_tensor, input_tensor[0], hn, cn)

            if i == 0:
                topv, topi = out.topk(10)
                topi = topi[0][random.randint(0, 9)]
            else:
                topv, topi = out.topk(2)
                topi = topi[0][random.randint(0, 1)]

            letter = codec.char_set[topi]
            if letter == codec.eos:
                break

            input_tensor = codec.encodeWord(letter)
            output_name += letter

        return output_name

    def initHidden(self):
        return torch.zeros(1, self.hiddensize)

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, k_catagories):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        # my model structure
        self.rnn_layer = nn.RNN(k_catagories+input_size,
                                hidden_size, 1)
        self.output = nn.Linear(k_catagories+hidden_size*2, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

        # Waterloo prof's structure
        # self.i2h = nn.Linear(k_catagories + input_size +
        #                      hidden_size, hidden_size)
        # self.i2o = nn.Linear(k_catagories + input_size +
        #                      hidden_size, output_size)
        # self.o2o = nn.Linear(hidden_size + output_size, output_size)
        # self.dropout = nn.Dropout(0.1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        # Waterloo prof
        # input_combined = torch.cat((category, input, hidden), 1)
        # hidden = self.i2h(input_combined)
        # output = self.i2o(input_combined)
        # output_combined = torch.cat((hidden, output), 1)
        # output = self.o2o(output_combined)
        # output = self.dropout(output)
        # output = self.softmax(output)

        # my forward
        input_combined = torch.cat((category, input), 1)
        output, hidden = self.rnn_layer(input_combined, hidden)
        output = self.output(torch.cat((category, output, hidden), 1))
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

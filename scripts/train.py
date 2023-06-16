import torch
import torch.nn as nn
from network import RNN
from dataloader import Data
from vcodec import Vcodec
import json

import matplotlib.pyplot as plt
import time
import math

PATH = '../data/names/*.txt'
IGNORE = {'1', '/', '\xa0', '\n'}


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(model: RNN, criterion, optimizer, category_tensor, input_tensor, target_tensor):
    target_tensor.unsqueeze_(-1)
    word_length = input_tensor.size(0)
    hidden = model.initHidden()

    loss = torch.Tensor([0])
    model.zero_grad()

    for i in range(word_length):
        # get each prediction and add to the loss at each time step
        output, hidden = model(category_tensor, input_tensor[i], hidden)
        L = criterion(output, target_tensor[i])
        loss += L

    # back propagation
    loss.backward()
    # opdate perameters
    optimizer.step()

    return output, loss.item()/input_tensor.size(0)


if __name__ == '__main__':
    start = time.time()

    n_iterations = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0
    learning_rate = 0.0005
    momentum = 0.9

    h_size = 256

    data = Data(PATH, IGNORE, sos='$', eos='&')
    k_categories = len(data.all_categories)
    c_characters = len(data.all_chars)

    model = RNN(c_characters, h_size, c_characters, k_categories)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum)

    codec = Vcodec(data.all_chars, data.all_categories)

    for iteration in range(1, n_iterations+1):
        category, word = data.randomTrainingSample()
        category_tensor = codec.catagoryTensor(category)
        word_tensor = codec.encodeWord(word)
        input_tensor, target_tensor = codec.inputTarget1DTensors(word_tensor)
        output, loss = train(model, criterion, optimizer,
                             category_tensor, input_tensor, target_tensor)
        total_loss += loss

        if iteration % print_every == 0:
            print('%s (%d %d%%) %.4f' %
                  (timeSince(start), iteration, iteration / n_iterations * 100, loss))

        if iteration % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    plt.figure()
    plt.plot(all_losses)
    plt.savefig('../model/losses.png')

    encoding_data = {}
    encoding_data["sos"] = '$'
    encoding_data["eos"] = '&'
    encoding_data["char_set"] = data.all_chars
    encoding_data["categories"] = data.all_categories
    encoding_data["h_size"] = h_size

    with open('../model/encoding-data.json', 'w', encoding='utf-8') as f:
        json.dump(encoding_data, f, ensure_ascii=False, indent=4)

    torch.save(model.state_dict(), '../model/Learned-weights.pth')
    print("...done")

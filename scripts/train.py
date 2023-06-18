import torch
import torch.nn as nn
from network import RNN
from network import LSTMModel
from dataloader import Data
from vcodec import Vcodec
import json

import matplotlib.pyplot as plt
import time
import math

PATH = '../data/names/*.txt'
IGNORE = {'1', '/', '\xa0', '\n', '\r', ' '}


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    start = time.time()

    n_iterations = 75000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0
    learning_rate = 0.0005
    momentum = 0.9

    h_size = 128

    data = Data(PATH, IGNORE, sos='$', eos='&')
    k_categories = len(data.all_categories)
    c_characters = len(data.all_chars)

    model = LSTMModel(c_characters, h_size, c_characters, k_categories)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum)

    codec = Vcodec(data.all_chars, data.all_categories)

    for iteration in range(1, n_iterations+1):
        category, word = data.randomTrainingSample()
        category_tensor = codec.catagoryTensor(category)
        word_tensor = codec.encodeWord(word)
        input_tensor, target_tensor = codec.inputTarget1DTensors(word_tensor)
        output, loss = model.trainWord(criterion, optimizer,
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

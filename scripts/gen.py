import torch
import torch.nn as nn
from network import RNN
from dataloader import Data
from vcodec import Vcodec
import json
import random


def sample(category, model, codec, char_set):

    max_length = 40
    output_name = ''
    input_tensor = codec.encodeWord('$')
    category_tensor = codec.catagoryTensor(category)
    hn = model.initHidden()

    for i in range(max_length):
        # print(f'Iteration number: {i}')
        out, hn = model(category_tensor, input_tensor[0], hn)

        p = torch.exp(out)

        topv, topi = out.topk(1)
        topi = topi[0][0]
        # print(f'topv: {topv}\ntopi: {topi}')

        letter = char_set[topi]
        # print(f'prediction: {letter}')
        if letter == eos:
            break

        input_tensor = codec.encodeWord(letter)
        output_name += letter

    print(f'category: {category}: {output_name}')

    return output_name


if __name__ == '__main__':

    f = open('../model/encoding-data.json', 'r', encoding='utf-8')

    encoding = json.load(f)

    sos = encoding['sos']
    eos = encoding['eos']
    categories = encoding['categories']
    char_set = encoding['char_set']
    h_size = encoding['h_size']
    N = len(char_set)
    K = len(categories)

    codec = Vcodec(char_set, categories)

    model = RNN(N, h_size, N, K)
    model.load_state_dict(torch.load('../model/learned-weights.pth'))
    model.eval()

    print(model)

    # current = ''
    # new = ''
    # while (current == new):
    #     new = sample('Scottish', model, codec)
    #     current = new
    #     print(new)

    sample('English', model, codec, char_set)
    sample('English', model, codec, char_set)
    sample('English', model, codec, char_set)
    sample('German', model, codec, char_set)
    sample('French', model, codec, char_set)
    sample('Japanese', model, codec, char_set)

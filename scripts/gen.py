from argparse import ArgumentParser, Namespace
import torch
from network import LSTMModel
from vcodec import Vcodec
import json


def generate(category, count=1):

    with open('../model/encoding-data.json', 'r', encoding='utf-8') as f:
        encoding = json.load(f)

    categories = encoding['categories']
    char_set = encoding['char_set']
    h_size = encoding['h_size']
    N = len(char_set)
    K = len(categories)

    codec = Vcodec(char_set, categories)

    model = LSTMModel(N, h_size, N, K)
    try:
        model.load_state_dict(torch.load('../model/learned-weights.pth'))
    except FileNotFoundError:
        print('A trained model was not found at location "/model/learned-weights.pth"')
    model.eval()

    samples = []
    for i in range(count):
        sample = model.sampleWord(category, codec)
        samples.append(sample)

    return samples


def stringify(l: list, direction=1, seperator=' '):
    output = '\n'
    for item in l:
        if direction == 1:
            output += f'{item}{seperator} '
        else:
            output += f'{item}{seperator}\n'

    return output


def main():
    with open('../model/encoding-data.json', 'r', encoding='utf-8') as f:
        encoding = json.load(f)

    categories = encoding['categories']

    parser = ArgumentParser(
        prog="Songell",
        description="Generates Fantasy RPG style names based on a character level LSTM model")

    parser.usage = "'category' [count]\n\nType -h for details"

    parser.add_argument("category", type=str,
                        help=f"The category to sample names from, model supports: {stringify(categories, seperator=',')}",)
    parser.add_argument("count", type=int, default=1, nargs='?',
                        help="The nummber of names to sample, default value is 1")

    args: Namespace = parser.parse_args()

    if args.category in categories:
        samples = generate(args.category, args.count)
        print(stringify(samples, direction=0, seperator=''))
    else:
        print(
            f'Songell error: argument "category" invalid choice. Choose from: {stringify(categories, direction=0, seperator="")}')


if __name__ == "__main__":
    main()

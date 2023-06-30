import torch
import json

from network import LSTMModel


def main():
    with open('../public/inferencing-model/encoding-data.json', 'r', encoding='utf-8') as f:
        encoding = json.load(f)

    categories = encoding['categories']
    char_set = encoding['char_set']
    h_size = encoding['h_size']
    N = len(char_set)
    K = len(categories)

    model = LSTMModel(N, h_size, N, K)

    try:
        model.load_state_dict(torch.load(
            '../public/inferencing-model/learned-weights.pth'))
    except FileNotFoundError:
        print('A trained model was not found at location "/public/inferencing-model/learned-weights.pth"')

    model.eval()
    category_tensor = torch.zeros(1, K)
    input_tensor = torch.zeros(1, N)
    hidden = torch.zeros(1, h_size)
    cell = torch.zeros(1, h_size)

    print(model)

    output = model(category_tensor, input_tensor, hidden, cell)
    print(f'testing model forward pass: {output}')

    torch.onnx.export(
        model,
        (category_tensor, input_tensor, hidden, cell),
        '../public/inferencing-model/learned-weights-onnx.onnx',
        export_params=True,
        input_names=['category', 'input', 'hiddenIN', 'cellIN'],
        output_names=['output', 'hiddenOUT', 'cellOUT'],
        verbose=True)

    print("...done")


if __name__ == '__main__':
    main()

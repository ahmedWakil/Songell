import torch


class Vcodec:

    def __init__(self, encoding_range, all_categories=[]):
        self.all_letters = encoding_range
        self.n_letters = len(encoding_range)
        self.category_map = self._buildCategoryMap(all_categories)

    def _buildCategoryMap(self, all_categories):
        category_map = {}
        for index, category in enumerate(all_categories):
            tensor = torch.zeros(1, len(all_categories))
            tensor[0][index] = 1
            category_map[category] = tensor
        return category_map

    def encodeWord(self, word):
        vectorized_word = torch.zeros(len(word), 1, self.n_letters)
        matrix_i = 0
        for letter in word:
            letter_index = self.all_letters.index(letter)
            vectorized_word[matrix_i][0][letter_index] = 1
            matrix_i += 1
        return vectorized_word

    def decodeWord(self, vectorized_word):
        word = ""
        for vector in vectorized_word:
            bool_mask = vector == 1
            index = bool_mask.nonzero(as_tuple=True)
            word = word + self.all_letters[index[1]]
        return word

    def catagoryTensor(self, category):
        return self.category_map[category]

    def inputTargetTensors(self, vectorized_word):
        input_tensor = vectorized_word[:-1][:][:]
        target_tensor = vectorized_word[1:][:][:]

        return (input_tensor, target_tensor)

    def inputTarget1DTensors(self, vectorized_word):
        input_tensor, target_tensor = self.inputTargetTensors(vectorized_word)
        target1D_tensor = torch.LongTensor(
            target_tensor.nonzero(as_tuple=True)[2])

        return input_tensor, target1D_tensor

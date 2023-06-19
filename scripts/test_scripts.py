import unittest
import torch
from dataloader import *
import vcodec
from network import RNN

R_TEST_ALL = '../data/test-names/*.txt'
R_TEST_ONE = "../data/test-names/Arabic.txt"
CHAR_TEST_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b',
                     'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                     'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALL_TEST_CATEGORY = ["Arabic", "English", "French", "German"]
TEST_IGNORE = {'1', '/', '\xa0', '\n', '\r'}

CATEGORY_DICT = {}
for i, category in enumerate(ALL_TEST_CATEGORY):
    tensor = torch.zeros(1, len(ALL_TEST_CATEGORY))
    tensor[0][i] = 1
    CATEGORY_DICT[category] = tensor

TEST_INPUT_SIZE = len(CHAR_TEST_LETTERS)
TEST_HIDDEN_SIZE = 64
TEST_OUTPUTSIZE = len(CHAR_TEST_LETTERS)


class Test(unittest.TestCase):

    def test_readLines_one(self):
        print("\nTEST: dataloader.readLines one...")
        correct = ["Khoury", "Nahas"]
        actual = readLines(R_TEST_ONE, TEST_IGNORE)
        msg = f'\nYour output: {actual}\nCorrect output: {correct}'
        self.assertEqual(correct, actual, msg)
        print(f'...passed')

    def test_readChars_one(self):
        print("\nTEST: dataloader.readChars one...")
        chars = "KhouryNahas"
        correct = set(chars)
        actual = readChars(R_TEST_ONE, TEST_IGNORE)
        msg = f'\nYour output: {actual}\nCorrect Ourput: {correct}'
        self.assertEqual(correct, actual, msg)
        print(f'...passed')

    def test_loadData_one(self):
        print("\nTEST: dataloader.loadData one...")
        correct_data = {}
        correct_data["Arabic"] = ["Khoury", "Nahas"]
        correct_catagories = ["Arabic"]
        chars = "KhouryNahas"
        correct_chars = sorted(list(set(chars)))

        actual_data, actual_categories, actual_chars = loadData(
            R_TEST_ONE, TEST_IGNORE)
        msg = f'Your data: {actual_data}\nCorect data:{correct_data}'
        self.assertEqual(correct_data, actual_data, msg)
        msg = f'Your categories: {actual_categories}\nCorrect categories: {correct_catagories}'
        self.assertEqual(correct_catagories, actual_categories)
        msg = f'Your chars: {actual_chars}\nCorrect chars: {correct_chars}'
        self.assertEqual(correct_chars, actual_chars)
        print(f'...passed')

    def test_loadData_two(self):
        print("\nTEST: dataloader.loadData two...")
        correct_data = {}
        correct_data["Arabic"] = ["Khoury", "Nahas"]
        correct_data["English"] = ["Yusuf", "Zaoui"]
        correct_data["French"] = [
            "Allard", "Beringer", "abcdefghijklmnopqrstuvwxyz"]
        correct_data["German"] = [
            "Abt", "Achilles", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

        actual_data, actual_categories, actual_chars = loadData(
            R_TEST_ALL, TEST_IGNORE)
        msg = f'\nYour data: {actual_data}\nCorect data:{correct_data}'
        self.assertEqual(correct_data, actual_data, msg)
        msg = f'\nYour categories: {actual_categories}\nCorrect categories: {ALL_TEST_CATEGORY}'
        self.assertEqual(ALL_TEST_CATEGORY, actual_categories, msg)
        msg = f'\nYour chars: {actual_chars}\nCorrect chars: {CHAR_TEST_LETTERS}'
        self.assertEqual(CHAR_TEST_LETTERS, actual_chars, msg)
        msg = f'\nTest output:\nData: {actual_data}\nCategories: {actual_categories}\nCharacters: {actual_chars}\nTotal number of chars: {len(actual_chars)}'
        print(msg)
        print(f'...passed')

    def test_encode_word_one(self):
        print("\nTEST: Vcodec encode one...")
        encoder = vcodec.Vcodec(CHAR_TEST_LETTERS)

        word = "test"
        correct_vectorized_word = torch.zeros(4, 1, len(CHAR_TEST_LETTERS))
        correct_vectorized_word[0][0][CHAR_TEST_LETTERS.index('t')] = 1
        correct_vectorized_word[1][0][CHAR_TEST_LETTERS.index('e')] = 1
        correct_vectorized_word[2][0][CHAR_TEST_LETTERS.index('s')] = 1
        correct_vectorized_word[3][0][CHAR_TEST_LETTERS.index('t')] = 1
        actual_vectorized_word = encoder.encodeWord(word)
        msg = f'Your vector:\n {actual_vectorized_word}\nCorrect vector:\n {correct_vectorized_word}'
        self.assertTrue(torch.equal(correct_vectorized_word,
                        actual_vectorized_word), msg)
        print(f'{msg}\n...passed')

    def test_decode_word_one(self):
        print("\nTEST: Vcodec decode one...")

        vectorized_word = torch.zeros(4, 1, len(CHAR_TEST_LETTERS))
        vectorized_word[0][0][CHAR_TEST_LETTERS.index('t')] = 1
        vectorized_word[1][0][CHAR_TEST_LETTERS.index('e')] = 1
        vectorized_word[2][0][CHAR_TEST_LETTERS.index('s')] = 1
        vectorized_word[3][0][CHAR_TEST_LETTERS.index('t')] = 1

        encoder = vcodec.Vcodec(CHAR_TEST_LETTERS)

        correct_word = "test"
        actual_word = encoder.decodeWord(vectorized_word)
        msg = f"Your word: {actual_word}\nCorrect word: {correct_word}"
        self.assertEqual(correct_word, actual_word, msg)
        print(f'{msg}\n...passed')

    def test_category_tensor_one(self):
        print("\nTEST: Vcodec categoryTensor one...")
        codec = vcodec.Vcodec(CHAR_TEST_LETTERS, ALL_TEST_CATEGORY)

        correct_vector = torch.zeros(1, len(ALL_TEST_CATEGORY))
        correct_vector[0][2] = 1
        actual_vector = codec.catagoryTensor("French")
        msg = f'\nYour category tensor: {actual_vector}\nCorrect category tensor: {correct_vector}'
        self.assertTrue(torch.equal(correct_vector, actual_vector), msg)
        print(f'\n...passed')

    def test_input_target_tensor_one(self):
        print("\nTEST: Vcodec InputTargetTensor one...")
        codec = vcodec.Vcodec(CHAR_TEST_LETTERS, ALL_TEST_CATEGORY)

        test_word = 'abcd'
        test_word_vectorized = codec.encodeWord(test_word)

        correct_input_tensor = torch.zeros(
            len(test_word)-1, 1, len(CHAR_TEST_LETTERS))
        correct_target_tensor = torch.zeros(
            len(test_word)-1, 1, len(CHAR_TEST_LETTERS))
        for i in range(0, len(test_word)-1):
            correct_input_tensor[i][0][CHAR_TEST_LETTERS.index(
                test_word[i])] = 1
            correct_target_tensor[i][0][CHAR_TEST_LETTERS.index(
                test_word[i+1])] = 1

        actual_input_tensor, actual_target_tensor = codec.inputTargetTensors(
            test_word_vectorized)
        msg = f'\nYour input: {actual_input_tensor}\nCorrect input: {correct_input_tensor}'
        self.assertTrue(torch.equal(correct_input_tensor,
                        actual_input_tensor), msg)
        print(msg)
        msg = f'\nYour target: {actual_target_tensor}\nCorrect target: {correct_target_tensor}'
        self.assertTrue(torch.equal(correct_target_tensor,
                        actual_target_tensor), msg)
        print(msg)
        print(f'...passed')

    def test_input_target_tensor_two(self):
        print("\nTEST: Vcodec InputTargetTensor two...")
        codec = vcodec.Vcodec(CHAR_TEST_LETTERS, ALL_TEST_CATEGORY)

        test_word = 'test'
        test_word_vectorized = codec.encodeWord(test_word)
        correct_input_tensor = torch.zeros(
            len(test_word)-1, 1, len(CHAR_TEST_LETTERS))
        correct_target_tensor = torch.zeros(
            len(test_word)-1, 1, len(CHAR_TEST_LETTERS))
        for i in range(0, len(test_word)-1):
            correct_input_tensor[i][0][CHAR_TEST_LETTERS.index(
                test_word[i])] = 1
            correct_target_tensor[i][0][CHAR_TEST_LETTERS.index(
                test_word[i+1])] = 1

        actual_input_tensor, actual_target_tensor = codec.inputTargetTensors(
            test_word_vectorized)

        correct_input_word = codec.decodeWord(correct_input_tensor)
        correct_target_word = codec.decodeWord(correct_target_tensor)
        actual_input_word = codec.decodeWord(actual_input_tensor)
        actual_target_word = codec.decodeWord(actual_target_tensor)

        msg = f'\nYour input word: {actual_input_word}\nCorrect input word: {actual_input_word}'
        self.assertEqual(correct_input_word, actual_input_word, msg)
        msg = f'\nYour target word: {actual_target_word}\nCorrect target word: {actual_target_word}'
        self.assertEqual(correct_target_word, actual_target_word, msg)

        msg = f'test word: test\nYour input/target word pair:\n{actual_input_word}\n{actual_target_word}'
        print(msg)
        print(f'...passed')

    def test_input_target1D_tensor(self):
        print("\nTEST: Vcodec InputTarget1DTensor one...")
        codec = vcodec.Vcodec(CHAR_TEST_LETTERS, ALL_TEST_CATEGORY)

        test_word = 'test'
        test_word_vectorized = codec.encodeWord(test_word)
        correct_input_tensor = torch.zeros(
            len(test_word)-1, 1, len(CHAR_TEST_LETTERS))
        for i in range(0, len(test_word)-1):
            correct_input_tensor[i][0][CHAR_TEST_LETTERS.index(
                test_word[i])] = 1

        correct_target1D_tensor = torch.LongTensor(len(test_word)-1)
        for i, letter in enumerate(test_word[1:]):
            correct_target1D_tensor[i] = CHAR_TEST_LETTERS.index(letter)

        actual_input, actual_target1D = codec.inputTarget1DTensors(
            test_word_vectorized)

        msg = f'\bYour input: {actual_input}\nCorrect input: {correct_input_tensor}'
        self.assertTrue(torch.equal(correct_input_tensor, actual_input), msg)
        print(msg)
        msg = f'\nYour 1Dtensor: {actual_target1D}\nCorrect 1Dtensor: {correct_target1D_tensor}'
        self.assertTrue(torch.equal(
            correct_target1D_tensor, actual_target1D), msg)
        print(msg)
        print('...passed')

    def test_network_init(self):
        print("\nTEST: RNN initialization...")
        codec = vcodec.Vcodec(CHAR_TEST_LETTERS, ALL_TEST_CATEGORY)
        input_tensor = codec.encodeWord("k")
        category_tensor = codec.catagoryTensor("English")

        rnn = RNN(TEST_INPUT_SIZE, TEST_HIDDEN_SIZE,
                  TEST_OUTPUTSIZE, len(ALL_TEST_CATEGORY))
        hidden = rnn.initHidden()
        o, h = rnn(category_tensor, input_tensor[0], hidden)
        print(
            f'\noutput (size):sum: {o} ({o.size()})')
        print(f'hidden (size): {h} ({h.size()})')

        self.assertEqual(TEST_INPUT_SIZE, o.size(
            1), 'Output vector should be the same length as input')
        self.assertEqual(TEST_HIDDEN_SIZE, h.size(
            1), 'Incorrect hidden vector size')
        print(f'\n...passed')

    def test_Data_class_characters(self):
        print("\nTEST: Data class character list...")
        correct_all_characters = CHAR_TEST_LETTERS + ['$', '&']

        data = Data(R_TEST_ALL, TEST_IGNORE)

        actual_all_characters = data.all_chars
        msg = f'\nYour character list: {actual_all_characters}\nCorrect character list: {correct_all_characters}'
        self.assertEqual(correct_all_characters, actual_all_characters, msg)
        print(msg)
        print(f'Total Characters: {len(actual_all_characters)}')
        print('...passed')

    def test_Data_class_ignore(self):
        print("\nTEST: Data class ignore list...")
        TEST_IGNORE.add('$')
        TEST_IGNORE.add('&')
        correct_ignore = TEST_IGNORE

        data = Data(R_TEST_ALL, TEST_IGNORE)
        actual_ignore = data.ignored_chars
        msg = f'\nYour ignored char list: {actual_ignore}\nCorrect ignore char list: {correct_ignore}'
        self.assertEqual(correct_ignore, actual_ignore, msg)
        print(msg)
        print('...passed')

    def test_Data_class_data_init(self):
        print('\nTEST: Data class initialization...')

        correct_data, correct_categories, correct_chars = loadData(
            R_TEST_ALL, TEST_IGNORE)
        correct_chars.append('$')
        correct_chars.append('&')

        data = Data(R_TEST_ALL, TEST_IGNORE)

        actual_data = data.data
        actual_categories = data.all_categories
        actual_chars = data.all_chars

        msg = f'\nYour data: {actual_data}\mCorrect data: {correct_data}'
        self.assertEqual(correct_data, actual_data, msg)
        msg = f'\nYour categories: {actual_categories}\nCorrect categories: {actual_categories}'
        self.assertEqual(correct_categories, actual_categories, msg)
        msg = f'\nYour chars: {actual_chars}\nCorrect chars: {correct_chars}'
        self.assertEqual(correct_chars, actual_chars, msg)
        msg = f'\nTest output:\nData \n{actual_data}\nCategories\n{actual_categories}\nChars\n{actual_chars}'
        print(msg)
        print('...passed')

    def test_network_model_integrity(self):
        print("\nTEST: RNN model integrity...")
        data = Data(R_TEST_ALL, TEST_IGNORE)
        codec = vcodec.Vcodec(data.all_chars, data.all_categories)
        N = len(data.all_chars)
        H = 128
        K = len(data.all_categories)
        network = RNN(N, H, N, K)

        input_tensor = codec.encodeWord(data.data["English"][0])[0]
        category_tensor = codec.catagoryTensor("English")
        hidden = network.initHidden()

        output, hidden = network(category_tensor, input_tensor, hidden)

        correct_output_length = len(CHAR_TEST_LETTERS) + 2
        self.assertEqual(correct_output_length, output.size(1),
                         f'Output: should be of size (1, {correct_output_length})')

        msg = f'\nOutput: {output}\nSize={output.size()}\nHiden: {hidden}\nSize={hidden.size()}'
        print(msg)
        msg = f'\nTest: input: {input_tensor}\nCategory: {category_tensor}\nHidden: {hidden}'
        print(msg)
        msg = f'\nCharacters: {data.all_chars}\nSize = {N}'
        print(msg)
        print(f'\n...passed')

    def test_Data_randomSampleNames_one(self):
        print("TEST: Data class randome sample names integrity...")
        data = Data('../data/names/*.txt', TEST_IGNORE)

        sample_catagories, sample_word = data.randomSampleNNames(
            "English", 5)

        print(
            f'Sample categories: {sample_catagories}\nSample_word: {sample_word}')

    def test_Data_randomTrainingSample_one(self):
        print("TEST: Data class random training sample names integrity...")
        data = Data('../data/names/*.txt', TEST_IGNORE)

        for i in range(10):
            sample_catagories, sample_word = data.randomTrainingSample()
            print(
                f'{i}th iteration\nSample categories: {sample_catagories} Sample_word: {sample_word}')


if __name__ == '__main__':
    unittest.main()

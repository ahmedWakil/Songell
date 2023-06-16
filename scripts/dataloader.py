import glob
import codecs
import os
import random


def findFiles(path):
    return glob.glob(path)


def readLines(filepath, ignore: set = {}):
    lines = []

    with codecs.open(filepath, encoding='utf-8') as f:
        for line in f:
            for i in ignore:
                line = line.replace(i, '')
            lines.append(line)

    return lines


def readChars(filepath, ignore: set = {}):
    chars = set()

    with codecs.open(filepath, encoding='utf-8') as f:
        for char in f.read():
            if char not in ignore:
                chars.add(char)

    return chars


def loadData(path, ignore: set = {}):
    data = {}
    all_catagories = []
    all_chars = set()
    for file_url in findFiles(path):
        char_set = readChars(file_url, ignore)
        names = readLines(file_url, ignore)
        catagory = os.path.splitext(os.path.basename(file_url))[0]
        data[catagory] = names
        all_catagories.append(catagory)
        all_chars = all_chars.union(char_set)

    return (data, all_catagories, sorted(list(all_chars)))


class Data():

    def __init__(self, path: str, ignore: set = {}, sos='$', eos='&'):
        self.path = path
        self.eos = eos
        self.sos = sos
        self.ignored_chars = ignore
        self.ignored_chars.update([sos, eos])
        self.data, self.all_categories, self.all_chars = loadData(path, ignore)
        self.all_chars.extend([sos, eos])

    def randomSampleNNames(self, category, n):
        words = random.choices(self.data[category], k=n)
        return category, words

    def randomTrainingSample(self):
        category = random.choice(self.all_categories)
        word = random.choice(self.data[category])
        word = self.sos+word+self.eos
        return category, word

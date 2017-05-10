import sys
import os
from os.path import isfile, join
from Tokenizer import Tokenizer
import timeit
from multiprocessing import Pool

t = Tokenizer(negate=False, html_special=False)
TOKENS = []

def tok(file):
    f = open(file, 'r', encoding='utf8')
    tokens = t.tokenize(f.read())
    f.close()
    return tokens

def st():
    FOLDERS = set(sys.argv[1:])
    for folder in FOLDERS:
        files = [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))]
        for file in files:
            tok(file)


def main():
    FOLDERS = set(sys.argv[1:])
    for folder in FOLDERS:
        files = [join(folder, f) for f in os.listdir(folder) if isfile(join(folder, f))]
        TOKENS.extend(p.map(tok, files))
        


if __name__ == '__main__':
    p = Pool(8)
    main()
    print(len(TOKENS))
    print(TOKENS[24999])

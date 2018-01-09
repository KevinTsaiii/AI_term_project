# encoding=utf8
import os
import re
from collections import Counter

def read_data(fname, count, word2idx):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    for line in lines:
        words.extend(line.split())

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    data = list()
    for line in lines:
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])

    print("Read %s words from %s" % (len(data), fname))
    return data


def lookfor(word2idx, word):
    return word2idx[word] if word in word2idx else word2idx['<UKN>']

def read_our_data(fname, count, word2idx):
    if os.path.isfile(fname):
        with open(fname, encoding='utf8') as f:
            lines = f.readlines()
    else:
        raise FileNotFoundError("[!] Data %s not found" % fname)

    assert(len(lines) % 22 == 0)
    
    # Test code! need to be deleted when actually using
#     lines = lines[:22*30000]

    n_question = len(lines) // 22
    print('Number of example in %s: %d' % (fname, n_question))

    contexts = []
    querys = []
    candidates = []
    answers = []
    for i in range(n_question):
        if i % 10000 == 0:
            print('Processing %d' % i)
        hey = lines[i * 22: (i+1) * 22]
        context = hey[:20]
        context = [[lookfor(word2idx, word) for word in sen.split()[1:]] for sen in context]
        contexts.append(context)

        last = hey[20]
        last = last.split('\t')
        querys.append([lookfor(word2idx, word) for word in last[0].split()[1:]])
        answers.append(lookfor(word2idx, last[1]))
        # last[2] is a empty string
        candidates.append([lookfor(word2idx, word) for word in last[3][:-1].split('|')])

    # TODO: Discard too long context
    # TODO: How to encode query?

    return {'contexts': contexts,
            'querys': querys,
            'candidates': candidates,
            'answers': answers}

def read_test_data(fname, word2idx):
    if os.path.isfile(fname):
        with open(fname, encoding='utf8') as f:
            lines = f.readlines()
    else:
        raise FileNotFoundError("[!] Data %s not found" % fname)
    if len(lines) % 22 == 21:
        lines.append('')
        
    assert(len(lines) % 22 == 0)
    
    n_question = len(lines) // 22
    print('Number of example in %s: %d' % (fname, n_question))
    
    contexts = []
    querys = []
    candidates = []
    
    for i in range(n_question):
        if i % 10000 == 0:
            print('Processing %d' % i)
        hey = lines[i * 22: (i+1) * 22]
        context = hey[:20]
        context = [sen.split()[1:] for sen in context]
        context = [
            [lookfor(word2idx, word) for word in sen]
            for sen in context
        ]
        contexts.append(context)

        last = hey[20]
        last = last.split('\t')
        querys.append([lookfor(word2idx, word) for word in last[0].split()[1:]])
        candidates.append([lookfor(word2idx, word) for word in last[1][:-1].split('|')])
        
    return {'contexts': contexts,
            'querys': querys,
            'candidates': candidates}
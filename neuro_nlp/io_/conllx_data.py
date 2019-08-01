__author__ = 'max'

import os.path
import random
import numpy as np
from .alphabet import Alphabet
from .logger import get_logger
from . import utils
import torch

# Special vocabulary symbols - we always put them at the start.
PAD = "_PAD"
PAD_CHAR = "_PAD_CHAR"
PAD_POS = "_PAD_POS"
PAD_NER = "_PAD_NER"
PAD_TYPE = "_<PAD>"
ROOT = "_ROOT"
ROOT_CHAR = "_ROOT_CHAR"
ROOT_POS = "_ROOT_POS"
ROOT_NER = "_ROOT_NER"
ROOT_TYPE = "_<ROOT>"
END = "_END"
END_POS = "_END_POS"
END_NER = "_END_NER"
END_TYPE = "_<END>"
END_CHAR = "_END_CHAR"
_START_VOCAB = [PAD, ROOT, END]

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140]

from .reader import CoNLLXReader


def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=50000, embedd_dict=None,
                     min_occurence=1, normalize_digits=True):
    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            # logger.info("Processing data: %s" % data_path)
            with open(data_path, 'r') as file:
                for line in file:
                    #line = line.decode('utf-8')
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')

                    #word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                    word = tokens[1]
                    pos = tokens[4]
                    ner = tokens[3]
                    arc_tag = tokens[7]

                    pos_alphabet.add(pos)
                    ner_alphabet.add(ner)
                    arc_alphabet.add(arc_tag)

                    if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                        vocab_set.add(word)
                        vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    pos_alphabet = Alphabet('pos')
    ner_alphabet = Alphabet('ner')
    arc_alphabet = Alphabet('arc')
    if True or not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        pos_alphabet.add(PAD_POS)
        ner_alphabet.add(PAD_NER)
        arc_alphabet.add(PAD_TYPE)

        pos_alphabet.add(ROOT_POS)
        ner_alphabet.add(ROOT_NER)
        arc_alphabet.add(ROOT_TYPE)

        pos_alphabet.add(END_POS)
        ner_alphabet.add(END_NER)
        arc_alphabet.add(END_TYPE)


        vocab = dict()
        with open(train_path, 'r') as file:
            for line in file:
                #line = line.decode('utf-8')
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split('\t')


                #word = utils.DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                word = tokens[1]
                pos = tokens[4]
                ner = tokens[3]
                arc_tag = tokens[7]

                pos_alphabet.add(pos)
                ner_alphabet.add(ner)
                arc_alphabet.add(arc_tag)

                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        vocab_list = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(vocab_list))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        ner_alphabet.save(alphabet_directory)
        arc_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        ner_alphabet.load(alphabet_directory)
        arc_alphabet.load(alphabet_directory)

    word_alphabet.close()
    pos_alphabet.close()
    ner_alphabet.close()
    arc_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("NER Alphabet Size: %d" % ner_alphabet.size())
    logger.info("Arcs Alphabet Size: %d" % arc_alphabet.size())
    return word_alphabet, pos_alphabet, pos_alphabet, ner_alphabet, arc_alphabet


def read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, ner_alphabet, arc_alphabet, max_size=None,
              normalize_digits=True, symbolic_root=False, symbolic_end=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = CoNLLXReader(source_path, word_alphabet, char_alphabet, pos_alphabet, ner_alphabet, arc_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids,  inst.pos_ids, inst.pos_ids,  inst.pos_ids, inst.heads, inst.arc_ids])
                max_len = len([char_seq for char_seq in inst.pos_ids])
                if max_char_length[bucket_id] < max_len:
                    max_char_length[bucket_id] = max_len
                break

        inst = reader.getNext(normalize_digits=normalize_digits, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length


def get_batch(data, batch_size, word_alphabet=None, unk_replace=0.0):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]
    char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    cid_inputs = np.empty([batch_size, bucket_length, char_length], dtype=np.int64)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    nid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    hid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)
    aid_inputs = np.empty([batch_size, bucket_length], dtype=np.int64)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)
    single = np.zeros([batch_size, bucket_length], dtype=np.int64)

    for b in range(batch_size):
        wids, cid_seqs, pids, nids, hids, aids = random.choice(data[bucket_id])

        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = PAD_ID_WORD
        for c, cids in enumerate(cid_seqs):
            cid_inputs[b, c, :len(cids)] = cids
            cid_inputs[b, c, len(cids):] = PAD_ID_CHAR
        cid_inputs[b, inst_size:, :] = PAD_ID_CHAR
        # pos ids
        pid_inputs[b, :inst_size] = pids
        pid_inputs[b, inst_size:] = PAD_ID_TAG
        # mer ids
        nid_inputs[b, :inst_size] = nids
        nid_inputs[b, inst_size:] = PAD_ID_TAG
        # arc ids
        aid_inputs[b, :inst_size] = aids
        aid_inputs[b, inst_size:] = PAD_ID_TAG
        # heads
        hid_inputs[b, :inst_size] = hids
        hid_inputs[b, inst_size:] = PAD_ID_TAG
        # masks
        masks[b, :inst_size] = 1.0

        if unk_replace:
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[b, j] = 1

    if unk_replace:
        noise = np.random.binomial(1, unk_replace, size=[batch_size, bucket_length])
        wid_inputs = wid_inputs * (1 - noise * single)

    return wid_inputs, cid_inputs, pid_inputs, nid_inputs, hid_inputs, aid_inputs, masks


def iterate_batch(data, batch_size, word_alphabet=None, unk_replace=0.0, shuffle=False):
    data, max_char_length = data
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size <= 0:
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        nid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        aid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, nids, hids, aids = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # ner ids
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_ID_TAG
            # arc ids
            aid_inputs[i, :inst_size] = aids
            aid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            if unk_replace:
                for j, wid in enumerate(wids):
                    if word_alphabet.is_singleton(wid):
                        single[i, j] = 1

        if unk_replace:
            noise = np.random.binomial(1, unk_replace, size=[bucket_size, bucket_length])
            wid_inputs = wid_inputs * (1 - noise * single)

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], cid_inputs[excerpt], pid_inputs[excerpt], nid_inputs[excerpt], hid_inputs[excerpt], \
                  aid_inputs[excerpt], masks[excerpt]


def read_data_to_variable(source_path, word_alphabet, char_alphabet, pos_alphabet, ner_alphabet, arc_alphabet, max_size=None,
                          normalize_digits=True, symbolic_root=False, symbolic_end=False,
                          use_gpu=False):
    data, max_char_length = read_data(source_path, word_alphabet, char_alphabet, pos_alphabet, ner_alphabet, arc_alphabet,
                                      max_size=max_size, normalize_digits=normalize_digits,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size <= 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(utils.MAX_CHAR_LENGTH, max_char_length[bucket_id] + utils.NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        nid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        aid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, cid_seqs, pids, nids, hids, aids = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # ner ids
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_ID_TAG
            # arc ids
            aid_inputs[i, :inst_size] = aids
            aid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = torch.LongTensor(wid_inputs)
        pos = torch.LongTensor(pid_inputs)
        heads = torch.LongTensor(hid_inputs)
        arc = torch.LongTensor(aid_inputs)
        masks = torch.FloatTensor(masks)
        single = torch.LongTensor(single)
        lengths = torch.LongTensor(lengths)
        if use_gpu:
            words = words.cuda()
            #chars = chars.cuda()
            pos = pos.cuda()
            #ner = ner.cuda()
            heads = heads.cuda()
            arc = arc.cuda()
            masks = masks.cuda()
            single = single.cuda()
            lengths = lengths.cuda()

        data_variable.append((words, pos, pos, pos, heads, arc, masks, single, lengths))

    return data_variable, bucket_sizes


def get_batch_variable(data, batch_size, unk_replace=0.0):
    data_variable, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])
    bucket_length = _buckets[bucket_id]

    words, chars, pos, ner, heads, arc, masks, single, lengths = data_variable[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]
    if words.is_cuda:
        index = index.cuda()

    words = words[index]
    if unk_replace:
        ones = single.data.new(batch_size, bucket_length).fill_(1)
        noise = masks.data.new(batch_size, bucket_length).bernoulli_(unk_replace).long()
        words = words * (ones - single[index] * noise)

    return words, chars[index], pos[index], ner[index], heads[index], arc[index], masks[index], lengths[index]


def iterate_batch_variable(data, batch_size, unk_replace=0.0, shuffle=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size <= 0:
            continue

        words, chars, pos, ner, heads, arc, masks, single, lengths = data_variable[bucket_id]
        if unk_replace:
            ones = single.data.new(bucket_size, bucket_length).fill_(1)
            noise = masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            if words.is_cuda:
                indices = indices.cuda()
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield words[excerpt], chars[excerpt], pos[excerpt], ner[excerpt], heads[excerpt], arc[excerpt], \
                  masks[excerpt], lengths[excerpt]

def calc_num_batches(data, batch_size):
    _, bucket_sizes = data
    bucket_sizes_mod_batch_size = [int(bucket_size / batch_size) + 1 if bucket_size > 0 else 0 for bucket_size in bucket_sizes]
    num_batches = sum(bucket_sizes_mod_batch_size)
    return num_batches

def iterate_batch_variable_rand_bucket_choosing(data, batch_size, unk_replace=0.0):
    data_variable, bucket_sizes = data
    indices_left = [set(np.arange(bucket_size)) for bucket_size in bucket_sizes]
    while sum(bucket_sizes) > 0:
        non_empty_buckets = [i for i,bucket_size in enumerate(bucket_sizes) if bucket_size > 0]
        bucket_id = np.random.choice(non_empty_buckets)
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]

        words, chars, pos, ner, heads, arc, masks, single, lengths = data_variable[bucket_id]
        min_batch_size = min(bucket_size, batch_size)
        indices = torch.LongTensor(np.random.choice(list(indices_left[bucket_id]), min_batch_size, replace=False))
        set_indices = set(indices.numpy())
        indices_left[bucket_id] = indices_left[bucket_id].difference(set_indices)
        if words.is_cuda:
            indices = indices.cuda()
        words = words[indices]
        if unk_replace:
            ones = single.data.new(min_batch_size, bucket_length).fill_(1)
            noise = masks.data.new(min_batch_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single[indices] * noise)
        bucket_sizes = [len(s) for s in indices_left]
        yield words, chars[indices], pos[indices], ner[indices], heads[indices], arc[indices], masks[indices], lengths[indices]
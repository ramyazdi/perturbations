#!/usr/bin/env python
import traceback
import os
import numpy as np

def precision_calc(gold, res):
    super_root = 'aSR_RSa'
    puncts = ["(", ")"]
    prec = 0
    length = 0
    for g_s, r_s in zip(gold, res):
        #walking on sentences,
        g_s.sort(key=lambda l: l[0])
        r_s.sort(key=lambda l: l[0])
        assert len(g_s) == len(r_s), 'sentences with different length \ngold:{0} \nres:{1}'.format(g_s, r_s)
        s_length = len(g_s)
        s_prec = 0
        for g_word_head, r_word_head in zip(g_s, r_s):
            # assert g_word_head[0] == r_word_head[0]

            if g_word_head[0] != r_word_head[0]:
                if (g_word_head[0] not in puncts) and (r_word_head[0] not in puncts):
                    raise Exception("Different Words, '{0}' in gold and '{1}' in pred".format(
                        g_word_head[0],
                        r_word_head[0]),
                    )

            if g_word_head[0] == super_root:
                s_length = s_length - 1
                continue

            if g_word_head[1] == r_word_head[1]:
                s_prec += 1
        prec += s_prec
        length += s_length
    prec = float(prec) / length
    return prec


def precision_calc_dont_sort(gold, res):
    super_root = 'aSR_RSa'
    puncts = ["(", ")"]
    prec = 0
    length = 0
    count = 0

    prec_root = 0
    for g_s, r_s in zip(gold, res):
        # walking on sentences,
        # g_s.sort(key=lambda l: l[0])
        # r_s.sort(key=lambda l: l[0])
        # assert len(g_s) == len(r_s), 'sentences with different length \ngold:{0} \nres:{1}'.format(g_s, r_s)
        s_length = len(g_s)
        s_prec = 0

        for g_word_head, r_word_head in zip(g_s, r_s):
            # assert g_word_head[0] == r_word_head[0]

            if g_word_head[0] != r_word_head[0]:
                if (g_word_head[0] not in puncts) and (r_word_head[0] not in puncts):
                    count += 1
                    raise Exception("Different Words, '{0}' in gold and '{1}' in pred".format(
                        g_word_head[0],
                        r_word_head[0]))

            if g_word_head[0] == super_root:
                s_length = s_length - 1
                continue

            if g_word_head[1] == r_word_head[1]:
                s_prec += 1
                if g_word_head[1]=='0':
                    prec_root += 1


        prec += s_prec
        length += s_length
    prec = float(prec) / length
    #print "root precision: "+str(float(prec_root) / len(gold))
    return prec

def fast_precision_calc(gold, res):
    #assuming the words are equal (not checking the words, only their heads)
    prec = 0
    length = 0
    for g_s, r_s in zip(gold, res):
        head_list_g_s = np.array(zip(*g_s)[1])
        head_list_r_s = np.array(zip(*r_s)[1])

        prec += sum(head_list_g_s==head_list_r_s)
        length += len(head_list_g_s)

    prec = float(prec) / length
    return prec

def fast_precision_calc_per_sentence(gold_sent, res_sent):
    #assuming the words are equal (not checking the words, only their heads)
    prec = 0
    length = 0

    head_list_g_s = np.array(zip(*gold_sent)[1])
    head_list_r_s = np.array(zip(*res_sent)[1])

    prec += sum(head_list_g_s==head_list_r_s)
    length += len(head_list_g_s)

    prec = float(prec) / length
    return prec

def precision_calc_wrap(f_gold, f_res, sort_me=False):
    if sort_me:
        return precision_calc(file2word_head(f_gold), file2word_head(f_res))
    else:
        return precision_calc_dont_sort(file2word_head(f_gold), file2word_head(f_res))


def file2word_head(file_):
    # return [[sentence_1],[sentence_2],..] where sentence_i=[(word,head),(word,head),(word,head),..]
    w_h = []
    w_h_s = []
    l_f = open(file_, 'r').readlines()
    for line in l_f:
        if not line.split():
            w_h.append(w_h_s)
            w_h_s = []
            continue
        s_l = line.split()
        word = s_l[1]
        head = s_l[6]
        w_h_s.append((word, head))
    return w_h



#print precision_calc_wrap(f_gold = '/home/ram/PycharmProjects/Master_Technion/ko.conllu' , f_res= '/home/ram/PycharmProjects/Master_Technion/final_results_mst_UAS_perturbated_k_100_MLN_mono_lingual_unsupervised_noise_learning_equal_weights_05_05/multiply_experiment_opt_sigma_1.6_mst')


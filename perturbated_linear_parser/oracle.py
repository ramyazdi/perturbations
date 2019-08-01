import os, re
import numpy as np
import csv
from edmonds_proj.precision_calc import precision_calc_dont_sort

def file2word_head(file_):
    w_h = []
    w_h_s = []
    l_f = open(file_, 'r').readlines()
    for line in l_f:
        if not line.split():
            w_h.append(tuple(w_h_s))
            w_h_s = []
            continue
        s_l = line.split()
        word = s_l[1]
        head = s_l[6]
        w_h_s.append((word, head))
    return w_h


def files2sentences(list_of_files):
    """
    Args:
        list_of_files:

    Returns:
        list, contains repeated sentences
        [[s1,s1', s1*], [s2, s2', s2*], [s3 .....], [s4 ...], ... ]
    """
    sentences = []
    f_list = [file2word_head(f) for f in list_of_files]
    f1 = f_list[0]
    for sentence in f1:
        sentences.append([sentence])

    #right now sentences=f1
    for corpus in f_list[1:]:
        for s, f_s in zip(sentences, corpus):
            s.append(f_s)

    return sentences


def file2sentences(file,k_trees=False):
    def get_word_head(s_line):
        id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel = s_line[:8]
        return (word, head)

    f = file[0]
    f = open(f, 'r').readlines()
    sentence = []
    rep_sentence = []
    count_repeated = 0
    mylist = []
    for line in f:
        match_ = re.match(r"(.+)(sent.)+(\d+)", line)
        if match_:
            if rep_sentence:
                mylist.append(rep_sentence)
                rep_sentence = []
                sentence = []
            count_repeated = 0
            continue
        if (k_trees and count_repeated>=k_trees): # control number of maximum trees to take (e.g. executed over liang k=500 but need k=100)
            continue

        if not line.split():
            if sentence:
                rep_sentence.append(tuple(sentence))
                sentence = []
                count_repeated += 1
            continue

        s_line = line.split()
        tup_unit = get_word_head(s_line)
        sentence.append(tup_unit)

    mylist.append(rep_sentence)
    return mylist

def accuracy_of_sentence(sentence, g_sentence):
    super_root = 'aSR_RSa'
    super_root_flag = False
    puncts = ["(", ")"]
    accuracy = 0
    for g_t, p_t in zip(g_sentence, sentence):
        # assert g_t[0] == p_t[0], "Different Words, {0} in gold and {1} in pred".format(g_t[0], p_t[0])
        if g_t[0] != p_t[0]:
            if (g_t[0] not in puncts) and (p_t[0] not in puncts):
                # print "sentence is {0}".format(sentence)
                # print "g_sentence is {0}".format(g_sentence)
                raise Exception("Different Words, {0} in gold and {1} in pred".format(g_t[0], p_t[0]))

        if g_t[0] == super_root:
            super_root_flag = True
            continue
        if g_t[1] == p_t[1]:
            accuracy += 1
    lensentence = len(g_sentence)-1 if super_root_flag else len(g_sentence)
    try:
        accuracy = float(accuracy) / lensentence
    except:
        print g_sentence
    return accuracy


def oracle(folder, gold):
    best_sentences = []
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    if len(files) > 1:

        sentences = files2sentences(files)
    elif len(files) == 1:

        sentences = file2sentences(files)
    gold_f = file2word_head(gold)
    for rep_sentence, gold_sentence in zip(sentences, gold_f):
        rep_sent_and_accuracy = []

        #make new list, take the rep sentence with highest accuracy.
        for sentence in rep_sentence:
            rep_sent_and_accuracy.append((accuracy_of_sentence(sentence, gold_sentence), sentence))

        rep_sent_and_accuracy = sorted(rep_sent_and_accuracy,
                                           key=lambda sent: sent[0], reverse=True)
        best_sentences.append(rep_sent_and_accuracy[0][1])
    _oracle = precision_calc_dont_sort(gold_f,best_sentences)#precision_calc(best_sentences, gold_f)
    return _oracle, best_sentences


def calculate_num_unique_trees(result_dir,noised_list_dir_name ='noised_list'):
    num_unique_trees_list = []
    for folder in os.listdir(result_dir):
        predicted_files_dir = os.path.join(result_dir,folder,noised_list_dir_name)
        files = [os.path.join(predicted_files_dir,file) for file in os.listdir(predicted_files_dir)][:250]

        if (len(files)==1): #kbest baseline model (sent1. 10 ,sent2. 20 etc..)
            filepath = files[0]
            for line in open(filepath, 'r').readlines():
                if line.split("\t") and re.match(r"(.+)(sent.)+(\d+)", line):
                    num_unique_trees_list.append(int(line.split("\t")[2]))
        elif (len(files)>1): #perturbated model
            sentences_all_files_list = files2sentences(files) #[[s1,s1', s1*..], [s2, s2', s2*..], [s3 .....], [s4 ...], ... ] where each s=((word,head),(word,head)(word,head)(word,head))
            for sentence_k_times in sentences_all_files_list:
                num_unique_trees_list.append(len(set(sentence_k_times)))
        else:
            raise Exception ("Directory "+predicted_files_dir+ ' is empty!')

    avg_unique_trees = np.mean(num_unique_trees_list)
    median_unique_trees = np.median(num_unique_trees_list)
    min_unique_trees = np.min(num_unique_trees_list)
    max_unique_trees = np.max(num_unique_trees_list)

    with open('../../statistics_over_trees_MLN_k_250.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['avg unique trees', 'median unique trees', 'min unique trees','max unique trees'])
        writer.writerow([avg_unique_trees,median_unique_trees,min_unique_trees,max_unique_trees])

if __name__== '__main__':
    #calculate_num_unique_trees(result_dir='../../DATA/results_k_minus_1_multiply_experiment_perturbated',noised_list_dir_name='noised_list_noise_learning_using_oracle')
    #n_dir = '/home/ram/PycharmProjects/Master_Technion/try_dir_cs'
    #gold ='/home/ram/PycharmProjects/Master_Technion/cs.conllu'
    _oracle, best_sentences = oracle('/home/ram/PycharmProjects/Master_Technion/noised_list_noise_MLN_K_100_noise_learning_unsupervised_equal_weights_05_05_id',
                                 '/home/ram/PycharmProjects/Master_Technion/id.conllu')
    print _oracle
    pass
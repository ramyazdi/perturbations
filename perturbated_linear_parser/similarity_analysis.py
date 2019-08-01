import os
import numpy as np
from utils import *
from oracle import *
from edmonds_proj.precision_calc import fast_precision_calc,precision_calc_dont_sort
import pickle
import ray
from globals import *


def calculate_range_of_uas(noised_list_dir,gold_file,is_distinct_list = False):
    files = [os.path.join(noised_list_dir, f) for f in os.listdir(noised_list_dir)]
    sentences = []
    if len(files) > 1:
        sentences = files2sentences(files[:100])
    elif len(files) == 1:
        sentences = file2sentences(files,k_trees=100)

    gold_f = file2word_head(gold_file)
    best_sentence_trees = []
    worst_sentence_trees = []
    percentile_25_sentence_trees,percentile_50_sentence_trees,percentile_75_sentence_trees = [],[],[]

    for rep_sentence, gold_sentence in zip(sentences, gold_f):
        rep_sent_and_accuracy = []

        if (is_distinct_list):
            rep_sentence = list(set(rep_sentence))
        #make new list, take the rep sentence with highest and lowest accuracy.
        for sentence in rep_sentence:
            rep_sent_and_accuracy.append((accuracy_of_sentence(sentence, gold_sentence), sentence))

        rep_sent_and_accuracy = sorted(rep_sent_and_accuracy,
                                           key=lambda sent: sent[0], reverse=False)

        rep_len = len(rep_sentence)
        index_25 = int(0.25*rep_len -1)
        index_50 = int(0.5 * rep_len - 1)
        index_75 = int(0.75 * rep_len - 1)
        percentile_25_sentence_trees.append(rep_sent_and_accuracy[index_25][1])
        percentile_50_sentence_trees.append(rep_sent_and_accuracy[index_50][1])
        percentile_75_sentence_trees.append(rep_sent_and_accuracy[index_75][1])

        best_sentence_trees.append(rep_sent_and_accuracy[-1][1])
        worst_sentence_trees.append(rep_sent_and_accuracy[0][1])

    best_uas = precision_calc_dont_sort(gold_f,best_sentence_trees)
    worst_uas = precision_calc_dont_sort(gold_f, worst_sentence_trees)

    percentile_25_uas = precision_calc_dont_sort(gold_f, percentile_25_sentence_trees)
    percentile_50_uas = precision_calc_dont_sort(gold_f, percentile_50_sentence_trees)
    percentile_75_uas = precision_calc_dont_sort(gold_f, percentile_75_sentence_trees)

    return best_uas, worst_uas,percentile_25_uas,percentile_50_uas,percentile_75_uas

def remove_all_trees_duplicated_less_than_k(trees_list,k=3):
    count_tree_list = [(trees_list.count(tree),tree) for tree in trees_list]
    if k>0:
        final_list = [count_tree_pair[1] for count_tree_pair in count_tree_list if count_tree_pair[0]>=k]
    else: #in case we want to remove all trees duplicated more than k
        k=abs(k)
        trees_to_remove = [count_tree_pair[1] for count_tree_pair in count_tree_list if count_tree_pair[0] > k]
        clipped_list = list(set([tuple(t) for t in trees_to_remove]))
        final_list = [count_tree_pair[1] for count_tree_pair in count_tree_list if count_tree_pair[0] <= k] +clipped_list
    return final_list

def calculate_hist_of_num_trees_per_accuracy(noised_list_dir,gold_file,threshold_duplicated_trees=None):
    files = [os.path.join(noised_list_dir, f) for f in os.listdir(noised_list_dir)]
    sentences = []
    if len(files) > 1:
        sentences = files2sentences(files[:100])
    elif len(files) == 1:
        sentences = file2sentences(files,k_trees=100)

    gold_f = file2word_head(gold_file)

    accuracy_sentences_dict =  {(np.round(k,1),np.round(k+0.1,1)):[] for k in np.arange(-0.1,1,0.1)} # accuracy:all sentences belong to the interval

    for rep_sentence, gold_sentence in zip(sentences, gold_f):

        for sentence in rep_sentence:
            acc_of_sent = accuracy_of_sentence(sentence, gold_sentence)
            accuracy_sentences_dict[[k for k in accuracy_sentences_dict if acc_of_sent>k[0] and acc_of_sent<=k[1]][0]].append(sentence)


    accuracy_num_of_trees_and_num_distinct_dict = {k:(len(accuracy_sentences_dict[k]),len(set(accuracy_sentences_dict[k])))
                                                   if threshold_duplicated_trees is None
                                                   else (len(remove_all_trees_duplicated_less_than_k(accuracy_sentences_dict[k]
                                                                                                      ,threshold_duplicated_trees)),
                                                         len(remove_all_trees_duplicated_less_than_k(list(set(accuracy_sentences_dict[k])),threshold_duplicated_trees))) for k in accuracy_sentences_dict}

    return accuracy_num_of_trees_and_num_distinct_dict


def analyze_best_tree(noised_list_dir,gold_file):
    files = [os.path.join(noised_list_dir, f) for f in os.listdir(noised_list_dir)]
    sentences = []
    if len(files) > 1:
        sentences = files2sentences(files[:100])
    elif len(files) == 1:
        sentences = file2sentences(files,k_trees=100)

    gold_f = file2word_head(gold_file)
    percentile_50_sentence_trees,percentile_75_sentence_trees = [],[]
    num_of_trees_with_best_acc_list, num_of_distinct_trees_with_best_acc_list = [],[]
    num_of_trees_with_worst_acc_list, num_of_distinct_trees_with_worst_acc_list = [], []
    nun_distinct_trees_list = []
    for rep_sentence, gold_sentence in zip(sentences, gold_f):
        rep_sent_and_accuracy = []

        #make new list, take the rep sentence with highest and lowest accuracy.
        for sentence in rep_sentence:
            rep_sent_and_accuracy.append((accuracy_of_sentence(sentence, gold_sentence), sentence))

        rep_sent_and_accuracy = sorted(rep_sent_and_accuracy,
                                           key=lambda sent: sent[0], reverse=False)

        best_tree_and_acc = rep_sent_and_accuracy[-1]
        trees_with_best_acc = [sent[1] for sent in rep_sent_and_accuracy if sent[0]==best_tree_and_acc[0]]
        num_of_trees_with_best_acc = len(trees_with_best_acc) #not all trees with accuracy = best_acc must be the chosen best tree
        num_of_distinct_trees_with_best_acc = len(set(trees_with_best_acc))
        num_of_trees_with_best_acc_list.append(num_of_trees_with_best_acc)
        num_of_distinct_trees_with_best_acc_list.append(num_of_distinct_trees_with_best_acc)


        worst_tree_and_acc = rep_sent_and_accuracy[0]
        trees_with_worst_acc = [sent[1] for sent in rep_sent_and_accuracy if sent[0]==worst_tree_and_acc[0]]
        num_of_trees_with_worst_acc = len(trees_with_worst_acc) #not all trees with accuracy = worst_acc must be the chosen worst tree
        num_of_distinct_trees_with_worst_acc = len(set(trees_with_worst_acc))
        num_of_trees_with_worst_acc_list.append(num_of_trees_with_worst_acc)
        num_of_distinct_trees_with_worst_acc_list.append(num_of_distinct_trees_with_worst_acc)


        nun_distinct_trees_list.append(len(set(rep_sentence)))

    return np.mean(num_of_trees_with_best_acc_list),np.median(num_of_trees_with_best_acc_list),\
           np.mean(num_of_distinct_trees_with_best_acc_list), np.median(num_of_distinct_trees_with_best_acc_list), \
           np.mean(num_of_trees_with_worst_acc_list), np.median(num_of_trees_with_worst_acc_list), \
           np.mean(num_of_distinct_trees_with_worst_acc_list), np.median(num_of_distinct_trees_with_worst_acc_list), \
           np.mean(nun_distinct_trees_list), np.median(nun_distinct_trees_list)



def calculate_similarity(k_files_list,gold_file):
    uas_list = []
    for res_file in k_files_list:
        uas = fast_precision_calc(gold_file, res_file)
        uas_list.append(uas)
    return uas_list

def calculate_similarity_per_sentence(same_sentence_chunk,gold_sentence):
    uas_list = []
    for res_sentence in same_sentence_chunk:
        uas = accuracy_of_sentence(res_sentence,gold_sentence)
        uas_list.append(uas)
    return uas_list

def calculate_similarity_for_kbest(sentences_list,distinct_trees= False):
    num_of_tokens_in_file = 0
    all_sentences_weighted_similarity = []
    for same_sentence_chunk in sentences_list:
        sent_len = len(same_sentence_chunk[0])
        num_of_tokens_in_file += sent_len
        uas_list_per_sentence = []
        if (distinct_trees):
            same_sentence_chunk = list(set(same_sentence_chunk))
        for target_sentence in same_sentence_chunk:
            uas_list_per_sentence += calculate_similarity_per_sentence(same_sentence_chunk,target_sentence)

        all_sentences_weighted_similarity.append(np.mean(uas_list_per_sentence)*sent_len)
    return sum(all_sentences_weighted_similarity)*1.0/num_of_tokens_in_file

def calculate_similarity_averaged_all_pairs(noised_list_dir,calculate_over_distinct_trees=False):
    files = [os.path.join(noised_list_dir, f) for f in os.listdir(noised_list_dir)]
    sentences = []
    if len(files) > 1:
        sentences = files2sentences(files[:100])
    elif len(files) == 1:
        sentences = file2sentences(files,k_trees=100)
        return calculate_similarity_for_kbest(sentences)

    if (calculate_over_distinct_trees):
        # in case of sitinct trees the number of duplication prt sentence is different so Kbest method for similarity must be used
        return calculate_similarity_for_kbest(sentences,distinct_trees=True)

    list_of_files = zip(*sentences)
    all_uas_list = []
    for file in list_of_files:
        all_uas_list+=calculate_similarity(list_of_files,file)

    return  np.mean(all_uas_list)

def min_sim_between_max_sim_to_1best(input_dir, liang_1best_file,weight_sim_1best = 0.5,weight_sim_between = 0.5):
    similarity_between = calculate_similarity_averaged_all_pairs(input_dir)

    noised_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)][:100]
    list_of_files = zip(*files2sentences(noised_files))
    similarity_to_1best = np.mean(calculate_similarity(list_of_files,file2word_head(liang_1best_file)))

    return weight_sim_1best*similarity_to_1best+ weight_sim_between*(1- similarity_between)

def calculate_avg_num_distinct_trees(sentences_list):
    num_unique_list = []
    number_of_times_with_duplications  = len(sentences_list[0])
    for same_sentence_chunk in sentences_list:
        num_unique_list.append(len(set(same_sentence_chunk)))

    return np.mean(num_unique_list),number_of_times_with_duplications


def min_sim_between_distinct_trees_max_sim_to_1best(input_dir, liang_1best_file,weight_sim_1best = 0.5,weight_sim_between = 0.5):

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
    sentences = files2sentences(files[:100])
    similarity_between_distinct = calculate_similarity_for_kbest(sentences,distinct_trees=True)

    avg_num_unique_trees, total_num_trees = calculate_avg_num_distinct_trees(sentences)

    noised_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)][:100]
    list_of_files = zip(*files2sentences(noised_files))
    similarity_to_1best = np.mean(calculate_similarity(list_of_files,file2word_head(liang_1best_file)))

    #return weight_sim_1best*similarity_to_1best+ weight_sim_between*(1- similarity_between_distinct) if (avg_num_unique_trees*1.0/total_num_trees)<=0.7 else 0
    return weight_sim_1best * similarity_to_1best + weight_sim_between * (1 - similarity_between_distinct) if (int(avg_num_unique_trees+1) < total_num_trees) else 0

@ray.remote
def calculate_similarity_and_range_per_corpus(file_name,corpus_name,noised_list_dir,gold_file):
    #################similarity###################################
    # similarity = calculate_similarity_averaged_all_pairs(noised_list_dir)
    # print 'finished similarity calc for file: '+ file_name
    # create_dir(os.path.join(DATA,file_name.split('_')[0]+'_similarity_analysis'),drop_if_exist=False)
    # with open(os.path.join(DATA,file_name.split('_')[0]+'_similarity_analysis',file_name+'_similarity_analysis.csv'), 'wb') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow([corpus_name,str(similarity)])
    #
    # #################similarity - distinct trees list###################################
    # similarity_unique_trees = calculate_similarity_averaged_all_pairs(noised_list_dir,calculate_over_distinct_trees=True)
    # print 'finished distinct trees similarity calc for file: ' + file_name
    # create_dir(os.path.join(DATA, file_name.split('_')[0] + '_distinct_trees_similarity_analysis'), drop_if_exist=False)
    # with open(os.path.join(DATA, file_name.split('_')[0] + '_distinct_trees_similarity_analysis',file_name + '_distinct_trees_similarity_analysis.csv'), 'wb') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow([corpus_name, str(similarity_unique_trees)])
    #
    # #################uas range###################################
    best_uas, worst_uas,percentile_25_uas,percentile_50_uas,percentile_75_uas = calculate_range_of_uas(noised_list_dir, gold_file,is_distinct_list=False)
    print 'finished range calc for file: '+ file_name
    create_dir(os.path.join(DATA, file_name.split('_')[0] + '_uas_range_analysis'), drop_if_exist=False)
    with open(os.path.join(DATA,file_name.split('_')[0]+'_uas_range_analysis',file_name+'_uas_range_analysis.csv'), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([corpus_name,str(worst_uas),str(best_uas),str(percentile_25_uas),str(percentile_50_uas),str(percentile_75_uas)])

    #################uas range distinct tree###################################
    # best_uas_distinct, worst_uas_distinct,percentile_25_uas_distinct,percentile_50_uas_distinct,percentile_75_uas_distinct = calculate_range_of_uas(noised_list_dir, gold_file,is_distinct_list=True)
    # print 'finished range calc distinct tree list for file: '+ file_name
    # create_dir(os.path.join(DATA, file_name.split('_')[0] + '_uas_range_analysis_distinct_trees'), drop_if_exist=False)
    # with open(os.path.join(DATA,file_name.split('_')[0]+'_uas_range_analysis_distinct_trees',file_name+'_uas_range_analysis.csv'), 'wb') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow([corpus_name,str(worst_uas_distinct),str(best_uas_distinct),str(percentile_25_uas_distinct),str(percentile_50_uas_distinct),str(percentile_75_uas_distinct)])

    #################best and worst tree analysis ###################################
    # num_of_trees_with_best_acc_mean, num_of_trees_with_best_acc_median,num_of_distinct_trees_with_best_acc_mean, num_of_distinct_trees_with_best_acc_median,num_of_trees_with_worst_acc_mean, num_of_trees_with_worst_acc_median,num_of_distinct_trees_with_worst_acc_mean, num_of_distinct_trees_with_worst_acc_median,nun_distinct_trees_list_mean,nun_distinct_trees_list_median = analyze_best_tree(noised_list_dir, gold_file)
    # print 'finished best tree analysis for file: ' + file_name
    # create_dir(os.path.join(DATA, file_name.split('_')[0] + '_best_tree_analysis'), drop_if_exist=False)
    # with open(os.path.join(DATA,file_name.split('_')[0]+'_best_tree_analysis',file_name+'_best_tree_analysis.csv'), 'wb') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',')
    #     writer.writerow([corpus_name,num_of_trees_with_best_acc_mean, num_of_trees_with_best_acc_median,num_of_distinct_trees_with_best_acc_mean, num_of_distinct_trees_with_best_acc_median,num_of_trees_with_worst_acc_mean, num_of_trees_with_worst_acc_median,num_of_distinct_trees_with_worst_acc_mean, num_of_distinct_trees_with_worst_acc_median,nun_distinct_trees_list_mean,nun_distinct_trees_list_median])

    ##################histogram#######################################
    # accuracy_num_of_trees_and_num_distinct_dict = calculate_hist_of_num_trees_per_accuracy(noised_list_dir, gold_file)
    # print 'finished histogram creation for file: ' + file_name
    # create_dir(os.path.join(DATA, file_name.split('_')[0] + '_histogram_accuracy_num_trees'), drop_if_exist=False)
    # with open(os.path.join(DATA, file_name.split('_')[0] + '_histogram_accuracy_num_trees',corpus_name+'_histogram_dict'), 'wb') as output:
    #     pickle.dump(accuracy_num_of_trees_and_num_distinct_dict, output)


if __name__== '__main__':
    results_dir = os.path.join(DATA,'results_cldp_after_predicted_pos')#'results_k_minus_1_multiply_experiment_perturbated')
    noised_dir_name = 'noised_list_after_predicted_pos_perturbated_model'
    file_prefix = 'Perturbated-predicted-pos_'

    mono_lingual_lngs = ['ar','he','it','ja','pt','zh'] #['ar','zh','nl','en','fr','de','he','ja','ko','pt','sl','vi','sv','ru','es']

    corpus_dirs = [os.path.join(results_dir,corpus) for corpus in  os.listdir(results_dir) if corpus.split("_")[0] in mono_lingual_lngs]

    ray.init()
    res = ray.get([calculate_similarity_and_range_per_corpus.remote(file_name = file_prefix+os.path.basename(corpus),
                                                                  corpus_name = os.path.basename(corpus),
                                                                  noised_list_dir = os.path.join(corpus,noised_dir_name),
                                                                  gold_file = os.path.join(corpus,'test_set',os.listdir(os.path.join(corpus,'test_set'))[0]))
                for corpus in corpus_dirs])

    # print analyze_best_tree('/home/ram/PycharmProjects/Master_Technion/noised_list_noise_MLN_K_100_noise_learning_unsupervised_equal_weights_05_05',
    #                              '/home/ram/PycharmProjects/Master_Technion/id.conllu')
    pass



#noised_dir = '/home/ram/PycharmProjects/Master_Technion/CLDP/noised_list_noise_learning_using_oracle'
#g_file = '/home/ram/PycharmProjects/Master_Technion/CLDP/en.conllu'
#best_uas,worst_uas = calculate_range_of_uas(noised_dir,g_file)
#print "range: "+str(worst_uas)+"-"+str(best_uas)
#print calculate_similarity_averaged_all_pairs(noised_dir)
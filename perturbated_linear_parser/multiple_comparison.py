import os
import numpy as np
from scipy import stats
from utils import *
from oracle import *
from edmonds_proj.precision_calc import fast_precision_calc,precision_calc_dont_sort,fast_precision_calc_per_sentence
import pickle
import ray
from globals import *
import pandas as pd

@ray.remote
def calculate_uas_per_sentence(file_name,corpus,final_files_dirs_list,gold_file,columns_to_replace=[]):

    dir_path = os.path.join(DATA,'mono_lingual_per_sentence_uas_all_methods')
    create_dir(dir_path, drop_if_exist=False)

    res_df = pd.DataFrame()
    senctence_list_gold = file2word_head(gold_file)

    for file in final_files_dirs_list:
        file_path = os.path.join(corpus,file , os.listdir(os.path.join(corpus, file))[0])
        sentence_list_method_t = file2word_head(file_path)
        uas_per_sentence_list = [fast_precision_calc_per_sentence(gold_sent = senctence_t_gold,res_sent=sentence_t_method_t) for senctence_t_gold,sentence_t_method_t in zip(senctence_list_gold,sentence_list_method_t)]

        res_df[file] = uas_per_sentence_list

    if columns_to_replace:
        res_df.columns = columns_to_replace
    res_df.to_csv(os.path.join(dir_path,file_name)+'.csv')


def concat_all_method_csv_with_kbest(all_methods_dir,kbest_dir,final_dir):
    create_dir(final_dir,drop_if_exist=False)
    for file in os.listdir(all_methods_dir):
        all_methods_file_path = os.path.join(all_methods_dir,file)
        kbest_file_path = os.path.join(kbest_dir,file)

        all_method_df = pd.read_csv(all_methods_file_path)
        kbest_file_df = pd.read_csv(kbest_file_path)


        pd.concat([all_method_df,kbest_file_df[['1best']]],axis=1)[['MLN','MFN','ALN','AFN','Kbest','1best']].to_csv(os.path.join(final_dir,file),index=False)

def calculate_pvalue_per_lng(per_sentence_uas_df,cols_a = ('MFN','AFN'),cols_b = ('Kbest','1best'),one_sided= True):
    pvalues_dict = {}
    for col_a in cols_a:
        for col_b in cols_b:
            t_results = stats.ttest_rel(per_sentence_uas_df[col_a], per_sentence_uas_df[col_b])
            pval = t_results[1]
            if (one_sided):
                # correct for one sided test
                pval = t_results[1]*1.0/ 2.0
                #check if t is negative
                if t_results[0]<0:
                    pval = 1-pval
            pvalues_dict[col_a+"_"+col_b] = [pval]

    return pd.DataFrame(pvalues_dict)


def calculate_pvals_for_all_lngs(per_senctence_uas_files_dir):
    pval_per_lng_df_list = []
    for file in os.listdir(per_senctence_uas_files_dir):
        file_path = os.path.join(per_senctence_uas_files_dir,file)
        per_sentence_uas_df = pd.read_csv(file_path)
        corpus_name = file.replace('cross_lingual_per_sentence_uas_','').replace('.csv','')
        pval_per_lng_df = calculate_pvalue_per_lng(per_sentence_uas_df)
        pval_per_lng_df['corpus'] = corpus_name

        pval_per_lng_df_list.append(pval_per_lng_df)

    pvals_all_lngs_df = pd.concat(pval_per_lng_df_list)
    pvals_all_lngs_df.to_csv('/home/ram/PycharmProjects/Master_Technion/mono_lingual_pvals_all_lngs_df.csv',index=False)
    return pvals_all_lngs_df

def find_k_estimator(pvalues, alpha, method ='B'):
    n = len(pvalues)
    pc_vec = [1]*n
    pvalues = sorted(pvalues, reverse = True)
    for u in range(0,n):
        if (u == 0):
            pc_vec[u] = calc_partial_cunjunction(pvalues, u+1, method)
        else:
            pc_vec[u] = max(calc_partial_cunjunction(pvalues, u+1, method), pc_vec[u - 1])
    k_hat = len([i for i in pc_vec if i<=alpha])
    return k_hat

def calc_partial_cunjunction(pvalues, u, method ='B'):
    n = len(pvalues)
    sorted_pvlas = pvalues[0:(n-u+1)]
    if (method == 'B'):
        p_u_n = (n-u+1)*min(sorted_pvlas)
    elif (method == 'F'):
        sum_chi_stat = 0
        for p in sorted_pvlas:
            sum_chi_stat = sum_chi_stat -2*np.log(p)
        p_u_n = 1-stats.chi2.cdf(sum_chi_stat,2*(n-u+1))

    return p_u_n

def perform_multiple_comparison(per_senctence_uas_files_dir,alpha = 0.05):
    pvals_all_lngs_df = calculate_pvals_for_all_lngs(per_senctence_uas_files_dir)
    for col in pvals_all_lngs_df.columns:
        if col.lower()!='corpus':
            print("comparison between: ",col)
            k = find_k_estimator(pvals_all_lngs_df[col], alpha, method= 'F')
            rejected_list = pvals_all_lngs_df.sort_values(by=col)[:k]['corpus'].tolist()
            not_rejected_list = pvals_all_lngs_df.sort_values(by=col)[k:]['corpus'].tolist()

            print(k," HO's were rejected")
            print ("Rejected list: ",rejected_list)
            print ("Not rejected list: ", not_rejected_list)
            print('-' * 50)
if __name__== '__main__':
    perform_multiple_comparison(per_senctence_uas_files_dir = '/home/ram/PycharmProjects/Master_Technion/mono_lingual_per_sentence_uas_all_methods')

    # concat_all_method_csv_with_kbest(all_methods_dir = '/home/ram/PycharmProjects/Master_Technion/per_sentence_uas_united',
    #                                  kbest_dir='/home/ram/PycharmProjects/Master_Technion/per_sentence_uas_1best',
    #                                  final_dir ='/home/ram/PycharmProjects/Master_Technion/per_sentence_uas_united_with_1best')
    # results_dir = os.path.join(DATA,'results_mono_lingual_experiment_perturbated')#'results_k_minus_1_multiply_experiment_perturbated')
    # # different methods' final list of trees
    # final_files_dirs = ['final_results_mst_UAS_perturbated_k_100_AFN_mono_lingual',
    #                     'final_results_mst_UAS_perturbated_k_100_MFN_mono_lingual',
    #                     'final_results_mst_UAS_liang_Kbest_baseline_k_100_mono_lingual',
    #                     'noised_list_1best']#['noised_list_1best'] #['final_results','final_results_mst_UAS_perturbated_k_100_MFN_fixed_noise','final_results_mst_UAS_perturbated_k_100_ALN','final_results_mst_UAS_perturbated_k_100_AFN']
    #
    # columns_to_replace = ['AFN','MFN','Kbest','1best']#['MLN','MFN','ALN','AFN']
    # file_prefix = 'mono_lingual_per_sentence_uas_'
    #
    # mono_lingual_lngs = ['ko','id','lv','el','da','fa','ur','et','hu','vi','cu','tr_pud','tr']
    #
    # corpora_to_exclude = ['ja','ja_pud','hi','hi_pud','ur']
    # corpus_dirs = [os.path.join(results_dir,corpus) for corpus in  mono_lingual_lngs]#os.listdir(results_dir) if corpus not in corpora_to_exclude]
    #
    # ray.init()
    # res = ray.get([calculate_uas_per_sentence.remote(file_name = file_prefix+os.path.basename(corpus),
    #                                           corpus = corpus,
    #                                           final_files_dirs_list  = final_files_dirs,
    #                                           gold_file = os.path.join(corpus,'test_set',os.listdir(os.path.join(corpus,'test_set'))[0]),
    #                                           columns_to_replace= columns_to_replace)
    #             for corpus in corpus_dirs])

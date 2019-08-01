#!/usr/bin/env python
import numpy as np

import os, tempfile, sys, shutil
import time
import mst_wrapper
from utils import *
from oracle import oracle
from parser_wrapper import ParserRunUnit
import logging
from globals import *
from similarity_analysis import *

log = logging.getLogger(__name__)
###########################noise learning#############################

def nultiple_files_assertion_error(_dir):
    str_ = "found {0} files instead of 1 in directory: \n{1}\n".format(len(os.listdir(_dir)),
                                                                       _dir)
    str_ += "files found: \n"
    for _file in os.listdir(_dir):
        str_ += os.path.join(_dir, _file)
        str_ += "\n"
    return str_

#######grid search using oracle#######
def optimal_noise_language(language, train_file, model_file, output_dir,eval_method = 'oracle',noise_method ='m',weight_sim_1best = 0.5,weight_sim_between = 0.5,noise_start =0,noise_end =2, noise_step =0.05):
    if (eval_method not in ['mst','oracle','unsupervised_min_sim_between_max_sim_to_1best','unsupervised_min_sim_between_distinct_trees_max_sim_to_1best']):
        raise Exception('evaluatiom method '+ eval_method +' is not implemnted')

    results_of_different_noises_path = os.path.join(output_dir, 'optimal_noises')
    print results_of_different_noises_path
    if (noise_method=='m'):
        mu = '1'
    else:
        mu = '0'

    tmp_train_file = tempfile.NamedTemporaryFile()
    convertconll2liang(train_file, tmp_train_file.name)

    if (eval_method in ['unsupervised_min_sim_between_max_sim_to_1best','unsupervised_min_sim_between_distinct_trees_max_sim_to_1best']): #create 1-best liang result for dev only once (not perturbated)
        tmp_out_file = tempfile.NamedTemporaryFile()
        pru = ParserRunUnit(language=language,
                            input_file=tmp_train_file.name,
                            model=model_file,
                            res_output=tmp_out_file.name
                            )

        pru.parse_no_words() #in practice - parse with words - model path
        # TODO:fix the file durig convertion and not before
        fix_liang_file(tmp_out_file.name)
        convertliang2conll(tmp_out_file.name, os.path.join(output_dir ,"liang_1_best_dev_res"))
        log.info('1best liang tree was created for unsupervised noise learning for language: '+language)

    for noise in np.arange(noise_start, noise_end, noise_step):
        specific_noise_dir = os.path.join(output_dir, "std_{0}_mu_{1}".format(noise, mu))

        create_dir(specific_noise_dir)

        output_file_path = os.path.basename(train_file)
        output_file_path = os.path.join(specific_noise_dir, output_file_path)

        for k in xrange(10):
            tmp_out_file = tempfile.NamedTemporaryFile()
            pru = ParserRunUnit(language=language,
                                input_file=tmp_train_file.name,
                                model=model_file,
                                res_output=tmp_out_file.name,
                                noise=True,
                                noise_method=noise_method,
                                mu=mu,
                                sigma=str(noise)
                                )

            pru.parse_no_words()
            #TODO:fix the file durig convertion and not before
            fix_liang_file(tmp_out_file.name)
            convertliang2conll(tmp_out_file.name, output_file_path + "_" + str(k))

        if (eval_method == 'oracle'):
            eval_score, best_list = oracle(specific_noise_dir, train_file)
        elif (eval_method == 'mst'):
            eval_score = mst_wrapper.mst_wrapper(input_dir=specific_noise_dir, gold_file=train_file, output_dir=specific_noise_dir)
        elif (eval_method == 'unsupervised_min_sim_between_max_sim_to_1best'):
            eval_score = min_sim_between_max_sim_to_1best(input_dir=specific_noise_dir, liang_1best_file = os.path.join(output_dir ,"liang_1_best_dev_res"),
                                                          weight_sim_1best = weight_sim_1best,weight_sim_between = weight_sim_between)

        elif (eval_method == 'unsupervised_min_sim_between_distinct_trees_max_sim_to_1best'):
            eval_score = min_sim_between_distinct_trees_max_sim_to_1best(input_dir=specific_noise_dir, liang_1best_file = os.path.join(output_dir ,"liang_1_best_dev_res"),
                                                          weight_sim_1best = weight_sim_1best,weight_sim_between = weight_sim_between)

        res = open(results_of_different_noises_path, 'a')
        res.write(str(noise) + " " + str(eval_score) + "\n")
        res.close()
        shutil.rmtree(specific_noise_dir)


def find_optimal_noise_per_language(eval_method = 'oracle',specific_languages=None,noise_method ='m',weight_sim_1best = 0.5,weight_sim_between = 0.5):

    language_dirs = [lng for lng in os.listdir(DATA) if lng.startswith("UD2")]

    if specific_languages:#executing over specific languages
        if (not isinstance(specific_languages,list)):
            specific_languages = [specific_languages]
        language_dirs = specific_languages

    for i,language in enumerate(language_dirs,1):

        print language
        language_dir = os.path.join(DATA,language)

        dev_file_name = [file for file in os.listdir(language_dir) if file.endswith("dev.conllu")][0]

        dev_set_path = os.path.join(language_dir, dev_file_name)

        model_path = os.path.join(language_dir,"Models","k_minus_1","model_"+language.split('_')[-1])

        output_dir = os.path.join(language_dir, "output_k_minus_1_experiment")
        create_dir(output_dir)

        ##test_file_name = [os.path.join(language_dir,"test",file) for file in os.listdir(os.path.join(language_dir,"test")) if file.endswith(".conllu")][0]

        optimal_noise_language(language, dev_set_path, model_path, output_dir,eval_method,noise_method,weight_sim_1best = weight_sim_1best,weight_sim_between = weight_sim_between)
        log.info('Noise learning finished for language: '+language+' ; '+str(i)+' out of '+str(len(language_dirs)) +'; using eval method: ' +eval_method)

if __name__ == "__main__":
    find_optimal_noise_per_language()


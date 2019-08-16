import train_parser as tp
import optimal_noise_k_minus_1 as optimal_n
import k_minus_1_noise_maker as noise_maker
import mst_wrapper as mst_wrapper
import oracle_wrapper as oracle_wrapper
import logging
import ray
import argparse

import utils
from globals import *

LOG_PATH = '../../'


class exec_process_parallel():

    def __init__(self, create_files = False,is_train_dev_together = False, log_file_name = LOG_PATH+'log_process'):
        self.create_files =  create_files
        self.log_file_name = log_file_name

        if (os.path.isfile(log_file_name)):
            os.remove(log_file_name)

        logging.basicConfig(filename=self.log_file_name,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        ray.init()

        logging.info('Execution started')
        if (self.create_files):
            logging.info('Strat creating per language files')
            utils.create_files_per_language_mono_lingual(num_sentences_per_lng_train = 300, num_sentences_per_lng_dev = 100, is_train_dev_together = is_train_dev_together) #all dev set
            logging.info('Finish creating per language files')

    @staticmethod
    @ray.remote
    def execute_parallel(language,eval_method='unsupervised_min_sim_between_max_sim_to_1best',train_models = False,find_noise = True,is_oracle_inference_results=True,fixed_noise=False,noise_method='m'):
        if (train_models):
            logging.info('-'*50)
            logging.info('Training process started')
            tp.train_parser_all_lng(language,is_train_dev_together,with_words=True)
            logging.info('Training process ended')

        if (find_noise and not fixed_noise):
            logging.info('-' * 50)
            logging.info('Optimal noise learning started')
            optimal_n.find_optimal_noise_per_language(eval_method=eval_method,specific_languages=language,noise_method=noise_method,weight_sim_1best = 0.5,weight_sim_between = 0.5)
            logging.info('Optimal noise learning ended')

        logging.info('-' * 50)
        logging.info('Noised dependency trees creation started')
        noise_maker.create_noised_dps_over_all_languages(language,is_train_dev_together= False,k_best_baseline=False,fixed_noise=fixed_noise,noise_method=noise_method)
        logging.info('Noised dependency trees creation ended')


        logging.info('-' * 50)
        logging.info('mst wrapper started')
        mst_wrapper.mst_wrapper_for_all_languages(language,
                                                  final_file_name='UAS_perturbated_k_100_MLN_FULL_mono_lingual',
                                                  is_train_dev_together=False,
                                                  is_mst=True,
                                                  given_one_file_repeated_sentnces=False)
        logging.info('mst wrapper ended')

        if (is_oracle_inference_results):
            logging.info('Oracle wrapper started')
            oracle_wrapper.oracle_wrapper_for_all_languages(language,final_file_name = 'UAS_perturbated_k_100_MLN_FULL_mono_lingual' ,is_train_dev_together=False)
            logging.info('Oracle wrapper ended')
        return language+" is ready"


    @staticmethod
    @ray.remote
    def execute_parallel_baseline_1_max_tree(language,train_models = False,is_train_dev_together=False):
        if (train_models):
            logging.info('-'*50)
            logging.info('Training process started')
            tp.train_parser_all_lng(language,is_train_dev_together,with_words=True)
            logging.info('Training process ended')

        logging.info('-' * 50)
        logging.info('Noised dependency trees creation started')
        noise_maker.create_noised_dps_over_all_languages(language,is_train_dev_together,mono_lingual=True)
        logging.info('Noised dependency trees creation ended')

        logging.info('-' * 50)
        logging.info('mst wrapper started')

        mst_wrapper.mst_wrapper_for_all_languages(language,
                                                  final_file_name='UAS_baseline_liang_1max_tree_FULL_mono_lingual',
                                                  is_train_dev_together=is_train_dev_together,
                                                  is_mst=False,
                                                  given_one_file_repeated_sentnces=False)
        logging.info('mst wrapper ended')

        return language+" is ready"



    @staticmethod
    @ray.remote
    def execute_parallel_baseline_k_best_trees(language,train_models = False,is_train_dev_together=False):
        if (train_models):
            logging.info('-'*50)
            logging.info('Training process started')
            tp.train_parser_all_lng(language,is_train_dev_together,with_words=True)
            logging.info('Training process ended')


        logging.info('-' * 50)
        logging.info('K-best (liang) dependency trees creation started')
        noise_maker.create_noised_dps_over_all_languages(specific_languages = language,
                                                         is_train_dev_together=is_train_dev_together,
                                                         k_best_baseline = 100)
        logging.info('K-best (liang) dependency trees creation ended')


        logging.info('-' * 50)
        logging.info('mst wrapper started')
        mst_wrapper.mst_wrapper_for_all_languages(language,
                                                  final_file_name='UAS_liang_Kbest_baseline_k_100_FULL_mono_lingual',
                                                  is_train_dev_together = is_train_dev_together,
                                                  is_mst=True,
                                                  given_one_file_repeated_sentnces=True)
        logging.info('mst wrapper ended')

        logging.info('-' * 50)
        logging.info('Oracle wrapper started')
        oracle_wrapper.oracle_wrapper_for_all_languages(language,final_file_name = 'UAS_liang_Kbest_baseline_k_100_FULL_mono_lingual' ,is_train_dev_together=is_train_dev_together)
        logging.info('Oracle wrapper ended')
        return language+" is ready"





parser = argparse.ArgumentParser()
parser.add_argument('--is_train_dev_together',action='store_true')
parser.add_argument('--baseline',default='no_baseline')
parser.add_argument('--create_files',action='store_true')
parser.add_argument('--train_models',action='store_true')
parser.add_argument('--dont_find_noise',action='store_true')
parser.add_argument('--eval_method',default='oracle')

parser.add_argument('--fixed_noise',default=False)
parser.add_argument('--noise_method',default='m')

args = parser.parse_args()

is_train_dev_together = args.is_train_dev_together
baseline = args.baseline #'k_best' ,None,'1_best'
create_files = args.create_files
train_models = args.train_models
fixed_noise = args.fixed_noise
noise_method = args.noise_method
find_noise = not (args.dont_find_noise)
eval_method = args.eval_method

exec_process_obj = exec_process_parallel(create_files=create_files,is_train_dev_together = is_train_dev_together)

mono_lingual_relevant_lngs = ['ar','zh','nl','en','fr','de','he','ja','ko','pt','sl','vi','sv','ru','es']

if (is_train_dev_together):
    model_dir_prefix = 'BASELINE_TRAIN_DEV_TOG_'
    language_dirs = [lng for lng in os.listdir(DATA) if lng.startswith(model_dir_prefix) and lng.split('_')[-1] in mono_lingual_relevant_lngs ]
else:
    model_dir_prefix = 'UD2_'
    language_dirs = [lng for lng in os.listdir(DATA) if lng.startswith(model_dir_prefix) and lng.split('_')[-1] in mono_lingual_relevant_lngs]

if (baseline =='1_best'):
    all_ready_lng = ray.get([exec_process_obj.execute_parallel_baseline_1_max_tree.remote(language,train_models=train_models,is_train_dev_together=is_train_dev_together) for language in language_dirs])

elif (baseline=='k_best'):
    all_ready_lng = ray.get([exec_process_obj.execute_parallel_baseline_k_best_trees.remote(language,train_models = train_models,is_train_dev_together=is_train_dev_together) for language in language_dirs])

elif (baseline=='no_baseline'):
    all_ready_lng = ray.get([exec_process_obj.execute_parallel.remote(language,
                                                               eval_method=eval_method,
                                                               train_models=train_models,
                                                               find_noise=find_noise,
                                                               is_oracle_inference_results=True,
                                                               fixed_noise=fixed_noise,
                                                               noise_method=noise_method) for language in language_dirs])
else:
    raise Exception("baseline "+baseline +" is not valid!")

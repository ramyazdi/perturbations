import tempfile
import os
import csv
from utils import   *
from edmonds_proj.noise_to_repeated_sentences import noise_to_repeated_sentences_temp_file
from edmonds_proj.wighted_graph_arc import file_list_2_wighted_graph
from edmonds_proj.precision_calc import precision_calc_wrap
import logging
from oracle import *
from globals import *

log = logging.getLogger(__name__)

def oracle_wrapper_for_all_languages(specific_languages=None,final_file_name = 'final_UAS_per_lng_k_minus_1' ,is_train_dev_together=False):

    if (is_train_dev_together):
        model_dir_prefix = 'BASELINE_TRAIN_DEV_TOG_'
    else:
        model_dir_prefix = 'UD2_'

    language_dirs = [lng for lng in os.listdir(RESULTS_DIR) if '~' not in lng]

    if specific_languages:#executing over specific languages
        if (not isinstance(specific_languages,list)):
            specific_languages = [specific_languages]

        specific_languages = [lng.replace(model_dir_prefix,"") for lng in specific_languages]
        language_dirs = [language_res_dir for language_res_dir in language_dirs  if [s_l for s_l in specific_languages if language_res_dir.startswith(s_l)]]

    with open(os.path.join(DATA,final_file_name+'_oracle_inference_'+str(language_dirs)+'.csv'), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i,language in enumerate(language_dirs,1):
            print language
            language_dir = os.path.join(RESULTS_DIR, language)

            gold_test_file = os.path.join(language_dir,'test_set',language.replace("_predicted_pos",'.conllu')+"_predicted_pos")#language+'.conllu')


            noised_files_dir = os.path.join(language_dir, NOISED_LIST_DIR_NAME)

            eval_score, best_list = oracle(noised_files_dir, gold_test_file)
            writer.writerow([language,str(eval_score)])

            log.info('Oracle calculation was finished for language ' + language + ' ; ' + str(i) + ' out of ' + str(len(language_dirs)))
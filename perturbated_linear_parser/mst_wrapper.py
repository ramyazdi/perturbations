import tempfile
import os
import csv
from utils import   *
from edmonds_proj.noise_to_repeated_sentences import noise_to_repeated_sentences_temp_file
from edmonds_proj.wighted_graph_arc import file_list_2_wighted_graph
from edmonds_proj.precision_calc import precision_calc_wrap
import logging
from globals import *

log = logging.getLogger(__name__)

def get_accuracy_file_vs_gold_file(a_file, gold_file):
    try:
        return precision_calc_wrap(gold_file, a_file)
    except Exception, e:
        print "Can't get accuracy of {0}\n{1}".format(a_file, str(e))
        return -1



def list_of_noised_files_2_mst(input_dir, output_dir,given_one_file_repeated_sentnces=False,remove_less_than_k_duplication=False):
    """
    :param input_dir: a directory, with multiple outputs (noised outputs)
    :param output_dir: will write a file, with regular output, made by first order mst into given output_dir
           file base_name will be similar to the input files (one of them)
    :param given_one_file_repeated_sentnces(bool): if True, get as input directory that
            contains only one file with repeated sentences (mostly when liang kbest is executed)
    :return: a file, mst of noised output
    """
    output_file_name = os.path.basename(os.listdir(input_dir)[0])
    output_file_name = '_'.join(output_file_name.split('_')[:-1])
    output_file_name += '_mst'
    output_file_path = os.path.join(output_dir, output_file_name)

    if (not given_one_file_repeated_sentnces):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        noise_to_repeated_sentences_temp_file(input_dir, temp_file,remove_less_than_k_duplication=remove_less_than_k_duplication)
        file_list_2_wighted_graph(temp_file.name, output_file_path)
    else:
        file_path = os.path.join(input_dir,os.listdir(input_dir)[0])
        file_list_2_wighted_graph(file_path, output_file_path)
    return output_file_path


def mst_wrapper(input_dir, gold_file, output_dir,is_mst = True,given_one_file_repeated_sentnces=False,remove_less_than_k_duplication=False):
    """

    :param input_dir: input_dir: a directory, with multiple outputs (noised outputs)
    :param gold_file: a conll gold_file
    :param output_dir: output_dir: will write a file, with regular output, made by first order mst into given output_dir
           file base_name will be similar to the input files (one of them)
    :return: accuracy
    """
    if is_mst:
        mst_file_path = list_of_noised_files_2_mst(input_dir, output_dir,given_one_file_repeated_sentnces,remove_less_than_k_duplication=remove_less_than_k_duplication)
    else:
        mst_file_path = [f for f in os.listdir(input_dir) if f.endswith("baseline_1_tree_model")][0]
        mst_file_path = os.path.join(input_dir,mst_file_path)
    print mst_file_path
    return get_accuracy_file_vs_gold_file(mst_file_path, gold_file)


def mst_wrapper_for_all_languages(specific_languages=None,final_file_name = 'final_UAS_per_lng_k_minus_1' ,is_train_dev_together=False,is_mst=True,given_one_file_repeated_sentnces = False,remove_less_than_k_duplication=False):

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

    with open(os.path.join(DATA,final_file_name+'_mst_inference_'+str(language_dirs)+'.csv'), 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for i,language in enumerate(language_dirs,1):
            print language
            language_dir = os.path.join(RESULTS_DIR, language)

            gold_test_file = os.path.join(language_dir,'test_set',language.replace("_predicted_pos",'.conllu')+"_predicted_pos")#language+'.conllu')

            noised_files_dir = os.path.join(language_dir, NOISED_LIST_DIR_NAME)

            final_result_dir = os.path.join(language_dir,'final_results_mst_'+final_file_name)
            create_dir(final_result_dir)

            final_accuracy = mst_wrapper(noised_files_dir, gold_test_file, final_result_dir,is_mst,given_one_file_repeated_sentnces,remove_less_than_k_duplication)
            writer.writerow([language,str(final_accuracy)])

            log.info('Aggregated dependency tree was created by MST wrapper for language ' + language + ' ; ' + str(i) + ' out of ' + str(len(language_dirs)))


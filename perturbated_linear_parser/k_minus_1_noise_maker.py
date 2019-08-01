#!/usr/bin/env python
import os, tempfile, sys
from utils import *
from parser_wrapper import ParserRunUnit
from shutil import copyfile
import logging
from globals import *

log = logging.getLogger(__name__)

my_code_dir = os.path.abspath(os.path.dirname(__file__))
final_parser_dir = os.path.dirname(my_code_dir)
proj_dir = os.path.dirname(final_parser_dir)

#####################return for test file the DP after the noising of weights #############
########### mult or additive to look on that######
###### k is const =100

def make_noised_list(opt_noise, language, test_file, output_dir, model,is_train_dev_together = False,noise_method ='m',mono_lingual=False,num_of_times=100):
    output_dir_path = os.path.join(output_dir,NOISED_LIST_DIR_NAME)#'noised_list_noise_learning_using_oracle')
    create_dir(output_dir_path)

    if (noise_method=='m'):
        mu = '1'
    else:
        mu = '0'

    output_file_prefix = os.path.join(output_dir_path,'multiply_experiment_opt_sigma_'+str(opt_noise))
    tmp_in_file = tempfile.NamedTemporaryFile()#os.path.join(output_dir_path,'ram_check2')#
    convertconll2liang(test_file, tmp_in_file.name)

    if (not is_train_dev_together and not mono_lingual):
        for k in xrange(num_of_times):
            tmp_out_file = tempfile.NamedTemporaryFile()
            pru = ParserRunUnit(language=language,
                                input_file=tmp_in_file.name,
                                model=model,
                                res_output=tmp_out_file.name,
                                noise=True,
                                noise_method=noise_method,
                                mu=mu,
                                sigma=opt_noise
                                )

            pru.parse_no_words()
            # TODO:fix the file durig convertion and not before
            fix_liang_file(tmp_out_file.name)
            convertliang2conll(tmp_out_file.name, output_file_prefix + "_" + str(k))
    else:
        log.info('Inference for baseline model 1 tree started')
        tmp_out_file = tempfile.NamedTemporaryFile()
        pru = ParserRunUnit(language=language,
                            input_file=tmp_in_file.name,
                            model=model,
                            res_output=tmp_out_file.name
                            )

        pru.parse_no_words()
        # TODO:fix the file durig convertion and not before
        fix_liang_file(tmp_out_file.name)
        convertliang2conll(tmp_out_file.name, output_file_prefix + "_baseline_1_tree_model")


def create_k_best_list(language, test_file, output_dir, model,k=100):
    output_dir_path = os.path.join(output_dir,NOISED_LIST_DIR_NAME)
    create_dir(output_dir_path)

    output_file_prefix = os.path.join(output_dir_path,'baseline_liang_'+str(k)+'_trees')
    tmp_in_file = tempfile.NamedTemporaryFile()#os.path.join(output_dir_path,'ram_check2')#
    convertconll2liang(test_file, tmp_in_file.name)


    log.info('Inference for baseline model '+str(k)+' trees started')
    tmp_out_file = tempfile.NamedTemporaryFile()
    pru = ParserRunUnit(language=language,
                        input_file=tmp_in_file.name,
                        model=model,
                        res_output=tmp_out_file.name
                        )

    pru.parse_no_words(k_best=str(k))
    fix_liang_file(tmp_out_file.name)
    convertliang2conll(tmp_out_file.name, output_file_prefix + "_baseline_k_trees_models")



def get_optimal_noise(language):

    noises_file_path = os.path.join(DATA, language,'output_k_minus_1_experiment', 'optimal_noises')

    print "reading noises file:\n{0}".format(noises_file_path)
    noise_oracle_list = open(noises_file_path, 'r').readlines()

    noise_oracle_map = {}
    for line in noise_oracle_list:
        key, val = line.strip().split(" ")
        noise_oracle_map[key] = float(val)

    max_oracle = max(noise_oracle_map.values())
    for noise in noise_oracle_map:
        if noise_oracle_map[noise] == max_oracle:
            print "Found optimal noise: {0}, Oracle={1}".format(noise, max_oracle)
            return noise
    raise Exception(
        "Can't find optimal noise\nmore detailed, max Oracle was {0}, Can't find optimal noise"
        "\nfile: {1}".format(
            max_oracle, noises_file_path)
    )
    return None


def create_noised_dps_over_all_languages(specific_languages = None,is_train_dev_together=False,k_best_baseline = False,fixed_noise = False,noise_method='m',mono_lingual=False):

    if (is_train_dev_together) :
        model_dir_prefix = 'BASELINE_TRAIN_DEV_TOG_'
    else:
        model_dir_prefix = 'UD2_'

    language_dirs = [lng for lng in os.listdir(DATA) if lng.startswith(model_dir_prefix)]

    if specific_languages:#executing over specific languages
        if (not isinstance(specific_languages,list)):
            specific_languages = [specific_languages]
        language_dirs = specific_languages

    for i,language in enumerate(language_dirs,1):
        print language
        language_dir_path = os.path.join(DATA, language)

        if (not is_train_dev_together):
            if (not fixed_noise):
                opt_noise = get_optimal_noise(language)
            else:
                opt_noise = fixed_noise # MFN perturbated model with fixed noise instead of learnable one
        else:
            opt_noise = '0'

        language_model_path = os.path.join(language_dir_path,'Models','k_minus_1',"model_"+language.split('_')[-1])

        test_files_per_lng = [test_file for test_file in os.listdir(os.path.join(language_dir_path,"test")) if '~' not in test_file and 'predicted_pos' in test_file]

        for t_file in test_files_per_lng:
            t_file_path = os.path.join(language_dir_path,'test', t_file)

            output_dir = os.path.join(RESULTS_DIR,t_file.replace(".conllu",''))
            create_dir(os.path.join(output_dir,'test_set'))

            copyfile(src = t_file_path, dst = os.path.join(output_dir,'test_set',t_file))

            if not k_best_baseline:
                make_noised_list(opt_noise, language, t_file_path, output_dir, language_model_path,is_train_dev_together,noise_method,mono_lingual = mono_lingual)

            else:
                create_k_best_list(language, t_file_path, output_dir, language_model_path,k_best_baseline)

        log.info('Noised list created per all tests file for language: ' + language + ' ; ' + str(i) + ' out of ' + str(len(language_dirs)))

#if __name__ == "__main__":
#    create_noised_dps_over_all_languages()

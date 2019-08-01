import subprocess

import ray
import torch
from utils_biaffine import *
import argparse

is_gpu_available = torch.cuda.is_available()
POS_TAGGER = 'python posTagger.py '

LOG_PATH = '../../'
DATA = '../../DATA'
UD2 = os.path.join(DATA,'Universal_Dependencies_2.0')


def execute_training_source_lng(language, num_epochs=200):
    model_path = os.path.join(DATA, language, 'pos_tagger')
    create_dir(model_path, drop_if_exist=False)
    langauge_dir_path = os.path.join(DATA, language)
    train_path = os.path.join(langauge_dir_path,
                              [file_l for file_l in os.listdir(langauge_dir_path) if file_l.endswith("train.conllu")][
                                  0])
    dev_path = os.path.join(langauge_dir_path,
                            [file_l for file_l in os.listdir(langauge_dir_path) if file_l.endswith("dev.conllu")][0])
    test_path = \
    [os.path.join(langauge_dir_path, "test", file_l) for file_l in os.listdir(os.path.join(langauge_dir_path, "test"))
     if '~' not in file_l and file_l.endswith('.conllu')][0]  # can be more than one file
    embedding_path = os.path.join(DATA,'bilingual_embedding', 'fasttext_embed', 'wiki.' + language.split('_')[-1] + '.vec')

    run_command = [POS_TAGGER, "--mode LSTM",
                   "--num_epochs " + str(num_epochs),
                   "--batch_size 16",
                   "--hidden_size 256",
                   "--num_layers 1",
                   "--char_dim 30",
                   # "--use_char" #should use only in monolingual
                   "--num_filters 30",
                   "--tag_space 256",
                   "--learning_rate 0.1",
                   "--decay_rate 0.05",
                   "--schedule 10",
                   "--gamma 0.0",
                   "--dropout std",
                   "--p_in 0.33",
                   "--p_rnn 0.33 0.5",
                   "--p_out 0.5",
                   "--unk_replace 0.0",
                   "--embedding fasttext",
                   '--embedding_dict ' + embedding_path,
                   '--train ' + train_path,
                   '--dev ' + dev_path,
                   '--test ' + test_path,
                   '--model_path ' + model_path,
                   '--model_name '+ "network_" + language + ".pt"]
    print (run_command)

    p = subprocess.Popen(" ".join(run_command), shell=True)
    p.communicate()
    return


class exec_process_parallel():

    def __init__(self, create_files = False, log_file_name = LOG_PATH+'log_process'):
        self.log_file_name = log_file_name

        if (os.path.isfile(log_file_name)):
            os.remove(log_file_name)

        log.basicConfig(filename=self.log_file_name,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=log.DEBUG)
        if (is_gpu_available):
            ray.init(num_gpus=2)
            print ("ray for gpu was initialized")
        else:
            ray.init()

        log.info('Execution started')
        if (create_files):
            log.info('Strat creating source language files')
            create_training_set_bilingual_pos(training_language_folder='UD_English')
            log.info('Finish creating source language files')


    @staticmethod
    @ray.remote(num_gpus=1)
    def execute_parallel_inference(source_language,target_language,inference_mode='regular_predict_file'):
        model_path = os.path.join(DATA, target_language, 'pos_tagger')
        create_dir(model_path, drop_if_exist=False)

        load_path = os.path.join(DATA,source_language,'pos_tagger',"network_"+source_language+".pt")

        langauge_dir_path = os.path.join(DATA,target_language)
        test_path_list = [os.path.join(langauge_dir_path,"test",file_l) for file_l in os.listdir(os.path.join(langauge_dir_path,"test")) if '~' not in file_l and file_l.endswith('.conllu')]#can be more than one file

        embedding_path = os.path.join(DATA,'bilingual_embedding','fasttext_embed','wiki.'+target_language.split('_')[-1]+'.vec')
        alignment_file_name = os.path.join(DATA,'bilingual_embedding','alignment_matrices',target_language.split("_")[-1]+'.txt')

        run_command =[POS_TAGGER, "--mode LSTM",
             "--num_epochs 10",
             "--batch_size 16",
             "--hidden_size 256",
             "--num_layers 1",
             "--char_dim 30",
             # "--use_char" #should use only in monolingual
             "--num_filters 30",
             "--tag_space 256",
             "--learning_rate 0.1",
             "--decay_rate 0.05",
             "--schedule 10",
             "--gamma 0.0",
             "--dropout std",
             "--p_in 0.33",
             "--p_rnn 0.33 0.5",
             "--p_out 0.5",
             "--unk_replace 0.0",
             "--embedding fasttext",
             '--embedding_dict '+embedding_path,
             '--train '+test_path_list[0],
             '--dev '+test_path_list[0],
             '--test '+','.join(test_path_list),
             '--load_path '+load_path,
             '--model_path '+model_path,
             '--alignment_file_name '+alignment_file_name,
             '--inference_mode ',inference_mode]

        p = subprocess.Popen(" ".join(run_command), shell=True)
        p.communicate()
        return


args_parser = argparse.ArgumentParser()

args_parser.add_argument('--create_files',action='store_true')
args = args_parser.parse_args()
create_files = args.create_files


exec_process_obj = exec_process_parallel(create_files=create_files)
execute_training_source_lng(language='POS_en',num_epochs=200)
print("Training step is over")

pos_experiment_lngs = ['zh','pt','it','ja','he','ar','da','fr','nl']

language_dirs = [lng for lng in os.listdir(DATA) if lng.startswith("UD2") and lng.split('_')[1]]

all_ready_lng = ray.get([exec_process_obj.execute_parallel_inference.remote(source_language='POS_en',target_language=language) for language in language_dirs])



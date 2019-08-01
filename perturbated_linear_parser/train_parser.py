#!/usr/bin/env python
import logging

from globals import *
from trainer_wrapper import TrainRunUnit
from utils import *

UD2 = os.path.join(DATA,UNIVERSAL_DEP_DIR)

log = logging.getLogger(__name__)


def train_parser_all_lng(specific_languages=None,is_train_dev_together=False,with_words = False):

    if (is_train_dev_together):
        model_dir_prefix = 'BASELINE_TRAIN_DEV_TOG_'
    else:
        model_dir_prefix = 'UD2_'

    language_dirs = [dir for dir in os.listdir(DATA) if dir.startswith(model_dir_prefix)]

    if specific_languages:#executing over specific languages
        if (not isinstance(specific_languages,list)):
            specific_languages = [specific_languages]
        language_dirs = specific_languages

    for i, language_dir in enumerate(language_dirs,1):

        log.info("Starting the training process for language: "+language_dir+" ; "+str(i)+' out of '+str(len(language_dirs)))

        language_path = os.path.join(DATA, language_dir)

        model_dir = os.path.join(language_path,'Models','k_minus_1')
        create_dir(model_dir,drop_if_exist=False)
        target_lng = language_dir.split('_')[-1]
        model_path = os.path.join(model_dir, 'model_'+target_lng)

        train_file_path = os.path.join(language_path, [file for file in os.listdir(language_path) if file.endswith("train.conllu")][0])

        tmp_train_file =  tempfile.NamedTemporaryFile() #os.path.join(language_path,'tem_train_file')
        convertconll2liang(train_file_path,tmp_train_file.name)
        tru = TrainRunUnit(target_lng, tmp_train_file.name, model_path)
        if (not with_words):
            tru.train_no_words()
        else:
            tru.train_with_words()
        log.info("Ending the training process for language: " + language_dir+" ; " + str(i) + ' out of ' + str(len(language_dirs)))

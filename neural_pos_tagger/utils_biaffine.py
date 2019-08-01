import logging as log
import os
import shutil
import pandas as pd

from neural_pos_tagger.exec_pos_tagger import UD2, DATA


def ud_file_2_sentences(filepath):
    lines = [i.strip() for i in open(filepath, 'r').readlines()]
    sentences = []
    sentence = []
    head_not_digit_flag = False
    for line in lines:
        index_col = line.split("\t")[0]
        if line.startswith('#') or '-' in index_col or '.' in index_col: #removing meta rows of ud , rows with 10-11,3-4 etc. and rows with 8.1,2.1...
            continue
        if not line.split():
            if (not head_not_digit_flag):
                sentences.append(sentence)

            head_not_digit_flag = False
            sentence = []
            continue

        splitted_line = line.split()
        splitted_line[1] = splitted_line[1].replace('(', '[').replace(')', ']')

        if (splitted_line[6].isdigit()):
            splitted_line[4],splitted_line[3] = splitted_line[3],splitted_line[4] #switch between UPOS to XPOS

        else:
            splitted_line[4], splitted_line[5] = splitted_line[5], splitted_line[4]
            splitted_line[6] = splitted_line[8]

        if (not splitted_line[6].isdigit() or splitted_line[4].isdigit()): #check head(6) and upos(4)
            head_not_digit_flag = True

        splitted_line[7] = 'NA'


        line = '\t'.join(splitted_line)

        sentence.append(line)
    if sentence:
        sentences.append(sentence)
    return sentences


def create_dir(dir,drop_if_exist=True):
    if os.path.isdir(dir):
        if not drop_if_exist:
            return
        shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir)

def sentences_2_conll_file(sentences, filepath,mode='w'):
    outfile = open(filepath,mode)
    for sentence in sentences:
        for line in sentence:
            outfile.write(line + "\n")
        outfile.write("\n")


def union_biaffine_files(b_dir = '../../biaffine_csv_files'):

    all_df_list =[]
    for file in os.listdir(b_dir):
        corpus= file.split('.conllu')[0]
        file_path = os.path.join(b_dir,file)
        lng_df = pd.read_csv(file_path,header=None)
        lng_df.columns = ['measurment','value']
        lng_df['corpus'] = corpus
        only_uas_df = lng_df[lng_df['measurment']=='UAS'][['corpus','value']]
        only_uas_df.rename(columns={'value':'UAS Biaffine'},inplace =True)
        only_uas_df['UAS Biaffine'] /=100.0
        only_uas_df['UAS Biaffine'] = only_uas_df['UAS Biaffine'].round(3)
        all_df_list.append(only_uas_df)
    final_df = pd.concat(all_df_list,axis=0)
    final_df.to_csv('../../biaffine_united_res.csv',index=False)


def copy_all_files_specific_suffix(from_dir,to_dir):
    for dir in os.listdir(from_dir):
        dir_path = os.path.join(from_dir,dir,'test')
        files_to_copy = [f for f in os.listdir(dir_path) if f.lower().endswith('.csv')]
        for file_to_copy in files_to_copy:
            shutil.copyfile(src=os.path.join(dir_path,file_to_copy),dst= os.path.join(to_dir,file_to_copy))


def create_training_set_bilingual_pos(training_language_folder='UD_English'):

    train_set_dir = os.path.join(UD2, 'ud-treebanks-conll2017')


    corpus_path_target = os.path.join(train_set_dir, training_language_folder)

    train_file_name_target = [file for file in os.listdir(corpus_path_target) if file.endswith("train.conllu")][0]
    train_file_path = os.path.join(corpus_path_target, train_file_name_target)
    list_train_lng = ud_file_2_sentences(train_file_path)

    dev_file_name_target = [file for file in os.listdir(corpus_path_target) if file.endswith("dev.conllu")]

    # handle with cases there is no dev set in the folder
    if (len(dev_file_name_target) > 0):
        dev_file_path = os.path.join(corpus_path_target, dev_file_name_target[0])
        list_dev_lng = ud_file_2_sentences(dev_file_path)
    else:
        list_dev_lng = []

    corpus_shorcut = train_file_name_target.split('-')[0]

    log.info("End reading train and dev files")


    # create a directory for training language
    model_dir = os.path.join(DATA, 'POS_'+corpus_shorcut)

    create_dir(model_dir, drop_if_exist=False)


    # save train set
    sentences_2_conll_file(sentences=list_train_lng,
                           filepath=os.path.join(model_dir,  corpus_shorcut+ '_train.conllu'),mode='w')

    # save dev set(contatins all other languages)
    dev_file_path = os.path.join(model_dir, corpus_shorcut + '_dev.conllu')
    sentences_2_conll_file(sentences=list_dev_lng, filepath=dev_file_path, mode='w')
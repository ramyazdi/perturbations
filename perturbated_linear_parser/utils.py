import random
import subprocess
import os
import shutil
import tempfile

import pandas as pd

from globals import DATA
from train_parser import UD2, log

this_file = __file__
my_code = os.path.dirname(this_file)
final_parser = os.path.dirname(my_code)
converter = os.path.join(final_parser,
                         'liang_convertor',
                         'liang_parser_converter-master',
                         'liangconverter.jar')


def print_dict_line_by_line(dict_):
    print "\n".join([" ".join((l, dict_[l])) for l in sorted(dict_.keys())])


def sentences_2_conll_file(sentences, filepath,mode='w'):
    outfile = open(filepath,mode)
    for sentence in sentences:
        for line in sentence:
            outfile.write(line + "\n")
        outfile.write("\n")


def sentences_2_conll_file_2(sentences, filepath,mode='w'):
    """
    won't leave the extra empty line
    """
    outfile = open(filepath, mode)
    sentence0 = sentences[0]
    for line in sentence0:
        outfile.write(line + "\n")
    for sentence in sentences[1:]:
        outfile.write("\n")
        for line in sentence:
            outfile.write(line + "\n")



def conll_file_2_sentences(filepath):
    lines = [i.strip() for i in open(filepath, 'r').readlines()]
    sentences = []
    sentence = []
    for line in lines:
        if not line.split():
            sentences.append(sentence)
            sentence = []
            continue
        sentence.append(line)
    if sentence:
        sentences.append(sentence)
    return sentences

def sentences_2_liang_file(sentences, filepath):
    outfile = open(filepath, 'w')
    for line in sentences:
        outfile.write(line)


def liang_file_2_sentences(filepath):
    lines = open(filepath, 'r').readlines()
    sentences = []
    for line in lines:
        sentences.append(line)
    return sentences


def convertliang2conll(orig, converted):
    command = "java -cp {0} converter.LiangTreeConverter l2c {1} {2}".format(
        converter, orig, converted
    )
    #print command
    p = subprocess.Popen(command, shell=True)
    p.communicate()


def convertconll2liang(orig, converted):

    command = "java -cp {0} converter.LiangTreeConverter c2l {1} {2}".format(
        converter, orig, converted
    )
    #print command
    p = subprocess.Popen(command, shell=True)
    p.communicate()


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

        line = '\t'.join(splitted_line)

        sentence.append(line)
    if sentence:
        sentences.append(sentence)
    return sentences



def list_str_2_file(list_str, filepath):

    outfile = open(filepath, 'w')
    for str in list_str:
        outfile.write(str + "\n")

def create_dir(dir,drop_if_exist=True):
    if os.path.isdir(dir):
        if not drop_if_exist:
            return
        shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir)

def fix_liang_file(filepath):
    lines = [i.strip() for i in open(filepath, 'r').readlines() if i.startswith("(")]
    fixed_file = open(filepath, 'w')
    for liang_str in lines:
        fixed_file.write(liang_str + "\n")


def check_num_of_sentences_and_tokens_in_ud_file(file_path):
    l_f = open(file_path, 'r').readlines()
    sentences_counter = 0
    tokens_counter = 0
    for line in l_f:
        if not line.split():
            sentences_counter +=1
        else:
            tokens_counter+=1
    print "file: ",os.path.basename(file_path)
    print "sentences_count: " ,sentences_counter
    print "tokens_count: ", tokens_counter

def union_csv_files(files_dir,output_csv_path,column_names = None):
    all_csv_dfs = [pd.read_csv(os.path.join(files_dir,file),header=None) for file in os.listdir(files_dir) if '~' not in file and "csv" in file.lower()]
    united_df = pd.concat(all_csv_dfs,axis=0)
    if column_names:
        united_df.columns = column_names

    united_df.to_csv(output_csv_path,index=False)

def merge_csv_with_summary_result(baseline_csv,summary_file = '../../../PREDICTED_POS_RANGE_summary.csv'):
    summary_df = pd.read_csv(summary_file)
    baseline_df = pd.read_csv(baseline_csv)
    final_df = summary_df.merge(baseline_df,how='left',on='corpus')
    final_path = os.path.join(os.path.dirname(summary_file),'PREDICTED_POS_RANGE_summary.csv')
    final_df.to_csv(final_path,index=False)


def create_files_per_language_cross_lingual(num_sentences_per_lng_train = 1000,num_sentences_per_lng_dev = 500,is_train_dev_together=False):
    # key: corpus name; value: [subset_of_train_set,subset_of_dev_set]
    lng_prefix_to_corpus = {}
    train_set_dir = os.path.join(UD2,'ud-treebanks-conll2017')
    log.info("start reading train and dev files for all languages")
    for corpus_dir_target in os.listdir(train_set_dir):
        corpus_path_target = os.path.join(train_set_dir,corpus_dir_target)

        #TODO: change x first sentences to sampling
        train_file_name_target = [file for file in os.listdir(corpus_path_target) if file.endswith("train.conllu")][0]
        train_file_path = os.path.join(corpus_path_target,train_file_name_target)
        list_train_lng = ud_file_2_sentences(train_file_path)[:num_sentences_per_lng_train]

        dev_file_name_target = [file for file in os.listdir(corpus_path_target) if file.endswith("dev.conllu")]

        #handle with cases there is no dev set in the folder
        if (len(dev_file_name_target)>0):
            dev_file_path= os.path.join(corpus_path_target,dev_file_name_target[0])
            list_dev_lng = ud_file_2_sentences(dev_file_path)[:num_sentences_per_lng_dev]
        else:
            list_dev_lng = []

        corpus_shorcut = train_file_name_target.split('-')[0]

        lng_prefix_to_corpus[corpus_shorcut] = [list_train_lng,list_dev_lng]

    log.info("End reading train and dev files for all languages")
    #all languages' prefix list
    lng_list = list(set([corpus.split('_')[0] for corpus in lng_prefix_to_corpus.keys()]))
    for index,target_lng in enumerate(lng_list,1):
        if (is_train_dev_together):
            model_dir_prefix = 'BASELINE_TRAIN_DEV_TOG_'
        else:
            model_dir_prefix = 'UD2_'
        #create a directory per language
        model_dir = os.path.join(DATA,model_dir_prefix+target_lng)

        create_dir(model_dir,drop_if_exist=False)

        all_other_lng_corpus = [(lng_prefix_to_corpus[source_corpus],source_corpus) for source_corpus in lng_prefix_to_corpus if source_corpus.split('_')[0]!= target_lng]

        all_other_lng_corpus_train = [train_dev_pair[0][0] for train_dev_pair in all_other_lng_corpus ]
        # concatinating all inner list into one flat list
        all_other_lng_corpus_train = [item for sublist in all_other_lng_corpus_train for item in sublist]

        all_other_lng_corpus_dev = [train_dev_pair[0][1] for train_dev_pair in all_other_lng_corpus if train_dev_pair[1]]
        all_other_lng_corpus_dev = [item for sublist in all_other_lng_corpus_dev for item in sublist]

        all_corpus_name_source = [train_dev_pair[1]  for train_dev_pair in all_other_lng_corpus]

        if (not is_train_dev_together):
            #save train set(contatins all other languages)
            sentences_2_conll_file(sentences=all_other_lng_corpus_train,filepath=os.path.join(model_dir,target_lng+'_all_other_lng_train.conllu'),mode='w')

            #save dev set(contatins all other languages)
            dev_file_path = os.path.join(model_dir,target_lng+'_all_other_lng_dev.conllu')
            sentences_2_conll_file(sentences=all_other_lng_corpus_dev,filepath=dev_file_path,mode='w')
            tmp_lng_dev = tempfile.NamedTemporaryFile()
            convertconll2liang(dev_file_path,tmp_lng_dev.name)
            os.remove(dev_file_path)
            convertliang2conll(tmp_lng_dev.name,dev_file_path)

            list_str_2_file(list_str=all_corpus_name_source,filepath=os.path.join(model_dir,target_lng+'_all_other_corpus_names.txt'))
        else:
            all_other_lng_corpus_train_dev_together = all_other_lng_corpus_dev+all_other_lng_corpus_train
            # save train and dev sets(contatins all other languages -) for baseline
            sentences_2_conll_file(sentences=all_other_lng_corpus_train_dev_together,filepath=os.path.join(model_dir, target_lng + '_all_other_lng_train.conllu'), mode='w')

        #reading test set per language and adding it into the language folder
        test_folder_path = os.path.join(UD2,'ud-test-v2.0-conll2017','gold','conll17-ud-test-2017-05-09')
        test_file_name_target_list = [file for file in os.listdir(test_folder_path) if file.startswith(target_lng)]

        create_dir(os.path.join(model_dir, 'test'))

        for file in test_file_name_target_list:
            test_file = ud_file_2_sentences(os.path.join(test_folder_path,file))
            test_file_path = os.path.join(model_dir,'test',file)
            sentences_2_conll_file(sentences=test_file,filepath=test_file_path)
            tmp_lng_test = tempfile.NamedTemporaryFile()
            convertconll2liang(test_file_path,tmp_lng_test.name)
            os.remove(test_file_path)
            convertliang2conll(tmp_lng_test.name,test_file_path)

        log.info("train, dev and test files were created for language: "+target_lng+" ; "+str(index)+' out of '+str(len(lng_list)))


def create_files_per_language_mono_lingual(num_sentences_per_lng_train=1000, num_sentences_per_lng_dev=500,is_train_dev_together=False):

    if (is_train_dev_together):
        model_dir_prefix = 'BASELINE_TRAIN_DEV_TOG_'
    else:
        model_dir_prefix = 'UD2_'


    # key: corpus name; value: [subset_of_train_set,subset_of_dev_set]
    lng_prefix_to_corpus = {}

    train_set_dir = os.path.join(UD2, 'ud-treebanks-conll2017')
    log.info("start reading train and dev files for all languages")
    for corpus_dir_target in os.listdir(train_set_dir):

        corpus_path_target = os.path.join(train_set_dir, corpus_dir_target)

        # TODO: change x first sentences to sampling
        train_file_name_target = [file for file in os.listdir(corpus_path_target) if file.endswith("train.conllu")][0]
        train_file_path = os.path.join(corpus_path_target, train_file_name_target)
        list_train_lng = ud_file_2_sentences(train_file_path)[:num_sentences_per_lng_train]

        dev_file_name_target = [file for file in os.listdir(corpus_path_target) if file.endswith("dev.conllu")]

        # handle with cases there is no dev set in the folder
        if (len(dev_file_name_target) > 0):
            dev_file_path = os.path.join(corpus_path_target, dev_file_name_target[0])
            list_dev_lng = ud_file_2_sentences(dev_file_path)[:num_sentences_per_lng_dev]
        else:
            list_dev_lng = []

        corpus_shorcut = train_file_name_target.split('-')[0]

        lng_prefix_to_corpus[corpus_shorcut] = [list_train_lng, list_dev_lng]

        log.info("End reading train and dev files for all languages")


    lng_list = list(set([corpus.split('_')[0] for corpus in lng_prefix_to_corpus.keys()]))

    for index, target_lng in enumerate(lng_list, 1):
        # create a directory per language
        model_dir = os.path.join(DATA, model_dir_prefix + target_lng)

        create_dir(model_dir, drop_if_exist=False)

        train_set_all_corpora = []
        dev_set_all_corpora = []
        for corpus_shorcut in lng_prefix_to_corpus:
            if corpus_shorcut.startswith(target_lng):
                train_set_all_corpora += lng_prefix_to_corpus[corpus_shorcut][0]
                dev_set_all_corpora += lng_prefix_to_corpus[corpus_shorcut][1]

        random.seed(123456) # for reproducibility
        random.shuffle(train_set_all_corpora)
        random.shuffle(dev_set_all_corpora)

        train_set_all_corpora = train_set_all_corpora[:num_sentences_per_lng_train]
        dev_set_all_corpora = dev_set_all_corpora[:num_sentences_per_lng_dev]

        if (not is_train_dev_together):
            # save train set(contatins all other languages)
            sentences_2_conll_file(sentences=train_set_all_corpora,
                                   filepath=os.path.join(model_dir, target_lng + '_train.conllu'),
                                   mode='w')

            # save dev set(contatins all other languages)
            dev_file_path = os.path.join(model_dir, target_lng + '_dev.conllu')
            sentences_2_conll_file(sentences=dev_set_all_corpora, filepath=dev_file_path, mode='w')
            tmp_lng_dev = tempfile.NamedTemporaryFile()
            convertconll2liang(dev_file_path, tmp_lng_dev.name)
            os.remove(dev_file_path)
            convertliang2conll(tmp_lng_dev.name, dev_file_path)

        else:
            train_dev_together = train_set_all_corpora + dev_set_all_corpora
            # save train and dev sets together for baselines
            sentences_2_conll_file(sentences=train_dev_together,
                                   filepath=os.path.join(model_dir, target_lng + '_together_train.conllu'),
                                   mode='w')

        # reading test set per language and adding it into the language folder
        test_folder_path = os.path.join(UD2, 'ud-test-v2.0-conll2017', 'gold', 'conll17-ud-test-2017-05-09')
        test_file_name_target_list = [file for file in os.listdir(test_folder_path) if file.startswith(target_lng)]

        create_dir(os.path.join(model_dir, 'test'))
        test_file_sizes ={}

        for file in test_file_name_target_list:
            test_file = ud_file_2_sentences(os.path.join(test_folder_path, file))
            test_file_sizes[file] = len(test_file)
            test_file_path = os.path.join(model_dir, 'test', file)
            sentences_2_conll_file(sentences=test_file, filepath=test_file_path)
            tmp_lng_test = tempfile.NamedTemporaryFile()
            convertconll2liang(test_file_path, tmp_lng_test.name)
            os.remove(test_file_path)
            convertliang2conll(tmp_lng_test.name, test_file_path)


        list_str_2_file(list_str=['train size', str(len(train_set_all_corpora)), 'dev size', str(len(dev_set_all_corpora)), 'test size', str(test_file_sizes)],
                        filepath=os.path.join(model_dir, target_lng + '_corpora_size.txt'))

        log.info("train, dev and test files were created for language: " + target_lng )
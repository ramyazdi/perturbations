__author__ = 'max'

import re
MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

import pandas as pd
import os
# Regular expressions used to normalize digits.
#DIGIT_RE = re.compile(br"\d")
DIGIT_RE = re.compile(r"\d")


def copy_all_result_files():
    import os
    from shutil import copyfile
    all_lngs = [os.path.join('DATA_BIAFFINE',lng,'test') for lng in os.listdir('DATA_BIAFFINE')]
    for lng in all_lngs:
        files = [f for f in os.listdir(lng) if f.endswith('final_all_lngs_updated_uas_more_alpha.csv')]
        for file_to_save in files:
            copyfile(os.path.join(lng,file_to_save),os.path.join('all_lngs_more_alpha',file_to_save))



def unite_initial_csv(csv_dir):
    all_csv_dfs = [(file.split('.')[0],pd.read_csv(os.path.join(csv_dir, file), header=None)) for file in os.listdir(csv_dir) if '~' not in file and "csv" in file.lower()]
    all_pds_T = []
    for df in all_csv_dfs:
        df_to_concat = df[1].set_index([0]).T
        df_to_concat['corpus'] = df[0]
        all_pds_T.append(df_to_concat)

    pd.concat(all_pds_T,axis=0).to_csv(os.path.join(os.path.dirname(csv_dir),'pos_kbest_with_majority_results.csv'))

#unite_initial_csv('/home/ram/PycharmProjects/Master_Technion/POS_baseline_kbest_with_majority')
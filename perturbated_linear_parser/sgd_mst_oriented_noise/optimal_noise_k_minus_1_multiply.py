#!/usr/bin/env python
import numpy as np

import os, tempfile, sys, shutil
import time
import random
import utils_sgd
from utils_sgd import epsilon, chunk_size, max_iter

mst_oriented_noise_dir = os.path.abspath(os.path.dirname(__file__))
my_code_dir = os.path.abspath(os.path.dirname(mst_oriented_noise_dir))
final_parser_dir = os.path.dirname(my_code_dir)
proj_dir = os.path.dirname(final_parser_dir)
sys.path.insert(0, my_code_dir)


from utils import convertconll2liang, convertliang2conll
from parser_wrapper import ParserRunUnit
import mst_wrapper
import utils


DATA = os.path.join(proj_dir, 'DATA')
OUTPUT_DIR = os.path.join(os.path.dirname(proj_dir),
                          'OUTPUT',
                          'stochastic_mst_oriented',
                          'multiply',
                          'k_minus_1',
                          )

BEST_NOISES_IN_LAST_METHOD = os.path.join(DATA,
                               'mst_oriented',
                               'multiply',
                               'k_minus_1',
                               'optimal_noises',
                               )


K_MINUS_1_MODELS_DIR = os.path.join(DATA, 'multiply_experiments/k_minus_1/dp_models')
TRIAN_FILES_DIR = os.path.join(DATA, "multiply_experiments/k_minus_1/train_noise")


def multiple_files_assertion_error(_dir):
    str_ = "found {0} files instead of 1 in directory: \n{1}\n".format(len(os.listdir(_dir)),
                                                                       _dir)
    str_ += "files found: \n"
    for _file in os.listdir(_dir):
        str_ += os.path.join(_dir, _file)
        str_ += "\n"
    return str_



def sgd_optimal_noise_language(language, train_file, model_file, output_dir):
    print language
    mu = '1'
    noise_method = 'm'
    mst_scores_history = [-2, -1]
    out_dir = os.path.join(output_dir, 'optimal_noise')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    best_results_path = os.path.join(out_dir, 'best_results')
    if not os.path.isdir(best_results_path):
        os.makedirs(best_results_path)

    train_file_name_basename = os.path.basename(train_file)
    convergence = False
    W, sign = utils_sgd.load_best_scores_and_vec(best_results_path)
    if sign:
        print "Found existed W"
    else:
        print "No existed W"
        vec_length = utils_sgd.get_len_of_features_vector(model_file)
        print vec_length
        W = get_v0(vec_length, language)
    train_file_sentences = utils.conll_file_2_sentences(train_file)
    step = 0

    while not convergence:
        step += 1
        eta = 1 / step
        chunk_indexes = random.sample(range(len(train_file_sentences)), chunk_size)
        chunk = [train_file_sentences[l] for l in chunk_indexes]
        chunk_train_file_conll = tempfile.NamedTemporaryFile()
        utils.sentences_2_conll_file_2(chunk, chunk_train_file_conll.name)
        mst_score_point = accuracy_of_mst(language,
                                          out_dir,
                                          W,
                                          chunk_train_file_conll,
                                          model_file,
                                          noise_method,
                                          mu,
                                          train_file_name_basename, )

        utils_sgd.save_score(best_results_path, step, mst_score_point, W)
        mst_scores_history.append(mst_score_point)
        for i, w in enumerate(W):
            w_aside = W[:]
            w_aside[i] = w + epsilon
            mst_aside_point = accuracy_of_mst(language,
                                              out_dir,
                                              w_aside,
                                              chunk_train_file_conll,
                                              model_file,
                                              noise_method,
                                              mu,
                                              train_file_name_basename,
                                              )
            # W[i] = w - eta * (mst_aside_point - mst_score_point) / epsilon
            W[i] = w - eta * (-mst_aside_point + mst_score_point) / epsilon  # sign changes are ment to help us find MAX
                                                                             #  instead of MIN,
                                                                             # as gradient descent naturally does.
        convergence = utils_sgd.check_covergence(mst_scores_history[-2], mst_scores_history[-1])
        if step >= max_iter:
            break



def accuracy_of_mst(language,
                    out_dir,
                    vector_noises_w,
                    chunk_train_file_conll,
                    language_model,
                    noise_method,
                    mu,
                    train_file_name_basename,
                    noise=0):
    orig_specific_noise_dir = os.path.join(out_dir, utils_sgd.random_gen())
    if not os.path.isdir(orig_specific_noise_dir):
        os.makedirs(orig_specific_noise_dir)

    noise_vec_dir = os.path.join(orig_specific_noise_dir, 'noise_vec')
    if not os.path.isdir(noise_vec_dir):
        os.makedirs(noise_vec_dir)

    specific_noise_dir = os.path.join(orig_specific_noise_dir, 'mst_input_files')
    if not os.path.isdir(specific_noise_dir):
        os.makedirs(specific_noise_dir)

    output_file_path = train_file_name_basename
    output_file_path = os.path.join(specific_noise_dir, output_file_path)
    noise_vec_path = os.path.join(noise_vec_dir, 'vec.npy')
    np.save(noise_vec_path, vector_noises_w)

    tmp_train_file_liang = tempfile.NamedTemporaryFile()
    convertconll2liang(chunk_train_file_conll.name, tmp_train_file_liang.name)
    tmp_out_file = tempfile.NamedTemporaryFile()
    pru = ParserRunUnit(language=language,
                        input_file=tmp_train_file_liang.name,
                        model=language_model,
                        res_output=tmp_out_file.name,
                        noise=True,
                        noise_method=noise_method,
                        mu=mu,
                        sigma=str(noise),
                        noise_file_path=noise_vec_path,
                        )
    print pru
    for k in xrange(10):
        tmp_out_file = tempfile.NamedTemporaryFile()
        pru.output = tmp_out_file.name
        pru.parse_no_words()
        convertliang2conll(tmp_out_file.name, output_file_path + "_" + str(k))
    mst_score = mst_wrapper.mst_wrapper(specific_noise_dir, chunk_train_file_conll.name, specific_noise_dir)
    shutil.rmtree(path=orig_specific_noise_dir)
    return mst_score






def get_v0(length, language):
    noises_path = os.path.join(BEST_NOISES_IN_LAST_METHOD,
                               language,
                               'optimal_noises',
                               )
    f_ = open(noises_path, 'r').readlines()
    noise_value_list = []
    for line in f_:
        if line:
            line = line.strip().split()
            noise_value_list.append((line[0], line[1]))
    noise_value_list.sort(key=lambda y: y[1], reverse=True)
    top_noise = noise_value_list[0][0]
    return np.ones(length) * float(top_noise)


def main():
    languages = os.listdir(TRIAN_FILES_DIR)
    for language in languages:
        print language
        train_file_dir = os.path.join(TRIAN_FILES_DIR, language)
        assert len(os.listdir(train_file_dir)) == 1, multiple_files_assertion_error(train_file_dir)
        train_file = os.path.join(train_file_dir, os.listdir(train_file_dir)[0])
        model_dir = os.path.join(K_MINUS_1_MODELS_DIR, language)
        assert len(os.listdir(model_dir)) == 1, multiple_files_assertion_error(model_dir)
        model_file = os.path.join(model_dir, os.listdir(model_dir)[0])
        output_dir = os.path.join(OUTPUT_DIR, language)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        sgd_optimal_noise_language(language, train_file, model_file, output_dir)


if __name__ == "__main__":
    main()


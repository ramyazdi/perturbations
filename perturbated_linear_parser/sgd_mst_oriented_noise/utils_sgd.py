import os
import numpy as np
import string
import random
epsilon = 1e-1
chunk_size = 100
max_iter = 100


def load_best_scores_and_vec(best_results_path):
    files = [os.path.join(best_results_path, l) for l in os.listdir(best_results_path)]
    if not files:
        return None, False
    scores_vec_tup = []
    for f_ in files:
        base_f = os.path.basename(f_)
        base_f = base_f.replace(".npy", "")
        score = float(base_f.split('_')[-1])
        scores_vec_tup.append((score, f_))
    scores_vec_tup.sort(key=lambda y: y[0], reverse=True)

    W = np.load(scores_vec_tup[0][-1])
    return W, True


def get_best_scores(best_results_path):
    files = [os.path.join(best_results_path, l) for l in os.listdir(best_results_path)]
    if not files:
        return []
    scores = []
    for f_ in files:
        base_f = os.path.basename(f_)
        base_f = base_f.replace(".npy", "")
        score = float(base_f.split('_')[-1])
        scores.append(float(score))
    return scores


def get_len_of_features_vector(path_to_file):
    f_model_lines = open(path_to_file, 'r').readlines()
    got_to_sign = False
    features = []

    for line in f_model_lines:
        l_s = line.strip()
        if not l_s:
            continue
        if l_s == "---":
            got_to_sign = True
            continue
        if got_to_sign:
            features.append(l_s)
    return len(features)


def random_gen(size=6):
    letters = string.ascii_uppercase + string.ascii_lowercase
    digits_letters = letters + string.digits
    first = random.choice(letters)
    rest = "".join(random.choice(digits_letters) for _ in range(size - 1))
    return first + rest


def save_score(best_results_path, step, mst, vec):
    existed_scors = get_best_scores(best_results_path)
    for score in existed_scors:
        if mst < score:
            return
    str_ = "_".join(('mst', str(step), random_gen(4), str(mst)))
    file_name = os.path.join(best_results_path, str_)
    np.save(file_name, vec)


def check_covergence(score1, score2):
    if abs(score1 - score2) < 1e-3:
        return True
    return False

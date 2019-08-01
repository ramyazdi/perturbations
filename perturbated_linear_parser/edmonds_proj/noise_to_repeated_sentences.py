import os
import edmonds_utils
from similarity_analysis import remove_all_trees_duplicated_less_than_k

def extra_line(num_of_sentences, sentence_index):
    return "1\tsent.{0}\t{1}\t_\t\t\t_\t0\t_\t_\t_\n\n".format(sentence_index, num_of_sentences)

def noise_to_repeated_sentences(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        new_dir = root.replace(input_folder, output_folder)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)

        sentencesContainer = []
        one_file_sentences = []
        sentence = []
        if files:
            for file_ in files:
                file_lines = open(os.path.join(root, file_), 'r').readlines()
                for line in file_lines:
                    if line.strip():
                        sentence.append(line)
                    else:
                        one_file_sentences.append(sentence)
                        sentence = []
                sentencesContainer.append(one_file_sentences)
                one_file_sentences = []
            rep_sentences = []
            ###
            one_file_sentences_ = sentencesContainer[0]
            for sentence in one_file_sentences_:
                rep_sentences.append([])
            ###
            for one_file_sentences_ in sentencesContainer:
                for index, sentence in enumerate(one_file_sentences_, 0):
                    try:
                        rep_sentences[index].append(sentence)
                    except:
                        # rep_sentences[index] = [sentence]
                        rep_sentences[index] = []
                        rep_sentences[index].append(sentence)

            filename = os.path.join(new_dir, files[0])
            output = open(filename, 'w')
            for index, one_rep_ in enumerate(rep_sentences, 1):
                number_of_rep = len(one_rep_)
                output.write(extra_line(number_of_rep, index))
                for sentence in one_rep_:
                    for line_ in sentence:
                        output.write(line_)
                    output.write("\n")
            output.close()


def noise_to_repeated_sentences_temp_file(input_folder, temp_file,remove_less_than_k_duplication=False):
    for root, dirs, files in os.walk(input_folder):
        sentencesContainer = []
        one_file_sentences = []
        sentence = []
        if files:
            for file_ in files[:100]:
                file_lines = open(os.path.join(root, file_), 'r').readlines()
                for line in file_lines:
                    if line.strip():
                        sentence.append(line)
                    else:
                        one_file_sentences.append(sentence)
                        sentence = []
                sentencesContainer.append(one_file_sentences)
                one_file_sentences = []
            rep_sentences = []
            ###
            one_file_sentences_ = sentencesContainer[0]
            for sentence in one_file_sentences_:
                rep_sentences.append([])
            ###
            for one_file_sentences_ in sentencesContainer:
                for index, sentence in enumerate(one_file_sentences_, 0):
                    try:
                        rep_sentences[index].append(sentence)
                    except:
                        # rep_sentences[index] = [sentence]
                        print index
                        rep_sentences[index] = []
                        rep_sentences[index].append(sentence)

            if (remove_less_than_k_duplication):
                for index,rep_sent in enumerate(rep_sentences, 0):
                    rep_sentences[index] = remove_all_trees_duplicated_less_than_k(rep_sent,k=remove_less_than_k_duplication)
                    if (len(rep_sentences[index])==0):
                        rep_sentences[index] = rep_sent


            filename = temp_file.name
            output = open(filename, 'w')
            for index, one_rep_ in enumerate(rep_sentences, 1):
                number_of_rep = len(one_rep_)
                output.write(extra_line(number_of_rep, index))
                for sentence in one_rep_:
                    for line_ in sentence:
                        output.write(line_)
                    output.write("\n")
            output.close()

def noise_to_repeated_sentences_2(input_folder, temp_file):
    files = [os.path.join(input_folder, l)for l in os.listdir(input_folder)]
    all_sentences = []
    length = len(edmonds_utils.conll_file_2_sentences(files[0]))
    print length
    for f_ in files:
        sentences = edmonds_utils.conll_file_2_sentences(f_)
        print len(sentences)/99
        all_sentences.append(sentences)
    transposed_senteces = []
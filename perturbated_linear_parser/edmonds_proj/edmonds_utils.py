import re
from collections import namedtuple

__all__ = ['Rep_Sentence_Statistics',
           'File_Statistics',
           'return_2',
           ]

NonWightedArc = namedtuple('Arc', ('tail', 'head'))
Arc = namedtuple('Arc', ('tail', 'weight', 'head'))


def return_2(x):
    return x


class Rep_Sentence_Statistics:
    def __init__(self, rep_sentence):
        self.rep_sentence = rep_sentence
        self.words = None
        self.edge_probability = {}

        self.get_words()
        self.number_of_rep_sentences = len(self.rep_sentence)
        self.number_of_words = self.get_length()
        self.number_of_unique_trees = self.unique_trees_counter()
        self.edge_probability_map()

    def my_is_equal(self, rss):
        """
        :param rss: Rep_Sentence_Statistics
        :return: boolean
        """
        return self.words == rss.words

    def get_words(self):
        words = []
        a_sentence = self.rep_sentence[0]
        for line in self.split_sentence(a_sentence):
            id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel = line
            words.append(word)
        self.words = words

    def edge_probability_map(self):
        edge_counter = {}
        counter = 0
        for sentence in self.rep_sentence:
            for line in self.split_sentence(sentence):
                id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel = line
                nw_arc = NonWightedArc(tail=id, head=head)
                if nw_arc not in edge_counter:
                    edge_counter[nw_arc] = 0
                edge_counter[nw_arc] += 1
            counter += 1
        edge_probability = {}
        for edge in edge_counter:
            edge_probability[edge] = float(edge_counter[edge]) / counter
        self.edge_probability = edge_probability

    def unique_trees_counter(self):
        trees_set = set()
        for sentence in self.rep_sentence:
            trees_set.add(str(self.sentence_to_tree_as_list(sentence)))
        return len(trees_set)

    def get_length(self):
        a_sentence = self.rep_sentence[0]
        return len(a_sentence)

    def sentence_to_tree(self, sentence):
        tree = []
        for line in self.split_sentence(sentence):
            id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel = line
            if head not in tree:
                tree[head] = []
            tree[head].append(id)
        return tree

    def sentence_to_tree_as_list(self, sentence):
        tree = []
        for line in self.split_sentence(sentence):
            id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel = line
            tree.append(NonWightedArc(tail=id, head=head))
        return sorted(tree)

    def split_sentence(self, sentence):
        splitted_sentence = []
        for line in sentence:
            s_line = line.split()
            id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel = s_line[:8]
            splitted_sentence.append([id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel])
        return splitted_sentence

    def print_graph(self, graph_):
        print "{"
        for key in graph_:
            print "'{0}': {1},".format(key, graph_[key])
        print "}"
        print "\n"


class File_Statistics:
    def __init__(self, _file):
        """
        :param file: data file with repeated sentences
        """
        self.rep_sentences_raw = file_2_repeated_sentences(_file)
        self.rep_sentences = []
        self.data = []
        self.make_rep_sentences_stats()

    def make_rep_sentences_stats(self):
        for rep_sentence in self.rep_sentences_raw:
            self.rep_sentences.append(Rep_Sentence_Statistics(rep_sentence))


def graph_to_inverse_values(graph_):
    minus_graph = {}
    for v in graph_:
        minus_graph[v] = {}
        for u in graph_[v]:
            minus_graph[v][u] = -graph_[v][u]
    return minus_graph


def dict_graph_2_arc_graph(dict_graph):
    """
    :param dict_graph: graph presented as a dictionary
    :return: graph presented as a list of Arcs

    in general, head points to tail
    """
    arcs_list = []
    for head in dict_graph:
        for tail in dict_graph[head]:
            arc = Arc(tail, dict_graph[head][tail], head)
            arcs_list.append(arc)
    return arcs_list


def arc_graph_2_dict_graph(arcs_list):
    """
    :param arcs_list: graph presented as a list of Arcs
    :return:  graph presented as a dictionary
    """
    dict_graph = {}
    for tail, weight, head in arcs_list:
        if head not in dict_graph:
            dict_graph[head] = {}
        dict_graph[head][tail] = weight
    return dict_graph


def file_2_repeated_sentences(_file):
    f_lines = open(_file, 'r').readlines()
    sentence = []
    one_repeated_sentences = []
    list_of_repeated_sentences = []
    for line in f_lines:
        if not line.strip():
            # sentence completed
            if sentence:
                one_repeated_sentences.append(sentence)
                sentence = []
                continue
            continue

        if re.findall(r'(sent.)+(\d)', line):
            # repeateds sentence completed
            if one_repeated_sentences:
                # if len(one_repeated_sentences) == 1:
                #     print line
                list_of_repeated_sentences.append(one_repeated_sentences)
                one_repeated_sentences = []
                continue
            continue

        sentence.append(line)
    if one_repeated_sentences:
        if sentence:
            one_repeated_sentences.append(sentence)
        list_of_repeated_sentences.append(one_repeated_sentences)
    return list_of_repeated_sentences


def file2sentences(file_):
    w_h = []
    w_h_s = []
    l_f = open(file_, 'r').readlines()
    for line in l_f:
        if not line.split():
            w_h.append(w_h_s)
            w_h_s = []
            continue
        w_h_s.append(line)
    return w_h


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
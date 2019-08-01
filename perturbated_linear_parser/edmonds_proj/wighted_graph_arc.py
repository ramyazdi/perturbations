import re, sys, os, precision_calc
import edmonds_utils
import second_order_mst
from athityakumar_mst import min_spanning_arborescence

sys.path.insert(0, './edmonds/edmonds')


languages = [
'arabic',
'basque',
'bulgarian',
'catalan',
'chinese',
'czech',
'danish',
'german',
'greek',
'hungarian',
'italian',
'japanese',
'portuguese',
'slovene',
'spanish',
'swedish',
]


class RepeatedSentence:
    """
    this class contains single graph wights.
    different trees of a single sentence will be used to make wights.

    graph = {'s':{'u':10, 'x':5},
     'u':{'v':1, 'x':2},
     'v':{'y':4},
     'x':{'u':3, 'v':9, 'y':2},
     'y':{'s':7, 'v':6}}

     head -----> dependent

    """
    def __init__(self, repeated_senteces):
        self.senteces = repeated_senteces
        self.graph = {}
        self.words = None
        self.artificial_root = '0'
        self.mst = None
        self.second_order_map = None
        self.conll_sentence = None

        #methods
        self.check_sentences()

        # self.make_wighted_graph()
        # self.mst_graph_edmonds()
        # self.mst2conll()


    def conll_sentence_2_str(self):
        s = ""
        for line in self.conll_sentence:
            s += "\t".join(line) + "\n"
        return s


    def mst2conll(self):
        """
        this dictionary
        {
        18 {'19': -44, '17': -34}
        0 {'2': -26, '21': -25}
        }
        says that 18 is the head of 19 and 17
        0 is head of 2 and 21
        (the numbers are weights)
        :return:
        """
        sentence = self.senteces[0]
        s_sentence = self.split_sentence(sentence)
        conll_sentence = []
        for id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel in s_sentence:
            for graph_head in self.mst:
                for graph_id in self.mst[graph_head]:
                    if graph_id == id:
                        conll_sentence.append([id, word, lemma, CPOSTAG, POSTAG, FEATS, graph_head, deprel])
        if len(conll_sentence) != len(self.words):
            print "\n\n\n"
            print "ERROR WITH MST2CONLL"
            print "length(mst_output) = {0}, lemngth(original_input)={1}".format(
                len(conll_sentence), len(self.words)
            )
            print "sentence created by MST algo:"
            print "\n".join(["\t".join(l) for l in conll_sentence])
            print "Original  sentence :"
            print "".join(sentence)
            print "MST Tree:"
            self.print_graph(self.mst)
            print "Original full graph"
            self.print_graph(self.graph)
        self.conll_sentence = conll_sentence


    def mst_second_order(self,alpha=None, beta=None):
        """
            NOTE:
                our graph is a dictionary, we need a list.
            """
        # inverse_values = self.graph_to_inverse_values(self.graph)
        list_graph = edmonds_utils.dict_graph_2_arc_graph(self.graph)
        s_o_mst_graph = second_order_mst.mst_2nd_order(list_graph,
                                                       len(self.words),
                                                       alpha,
                                                       beta,
                                                       second_order_map=self.second_order_map,
                                                       )
        self.mst = edmonds_utils.arc_graph_2_dict_graph(s_o_mst_graph)


    def mst_graph_edmonds(self):
        print "there is a bug, don't use"



    def mst_graph_athityakumr(self):
        arc_graph = edmonds_utils.dict_graph_2_arc_graph(self.graph_to_inverse_values(self.graph))
        spanning_arborescence = min_spanning_arborescence(arc_graph, self.artificial_root)
        spanning_arborescence_list = []
        for key in spanning_arborescence:
            spanning_arborescence_list.append(spanning_arborescence[key])
        self.mst = edmonds_utils.arc_graph_2_dict_graph(spanning_arborescence_list)


    def print_graph(self, graph_):
        print 'wighted graph:'
        print "{"
        for key in graph_:
            print "'{0}': {1},".format(key, graph_[key])
        print "}"
        print "\n"


    def make_second_order_wights(self):
        my_map = {}
        for sentence in self.senteces:
            s_sentence = self.split_sentence(sentence)
            for a_line in s_sentence:
                a_id, a_word, a_lemma, a_CPOSTAG, a_POSTAG, a_FEATS, a_head, a_deprel = a_line
                for b_line in s_sentence:
                    b_id, b_word, b_lemma, b_CPOSTAG, b_POSTAG, b_FEATS, b_head, b_deprel = b_line
                    if (a_id == b_head and a_head != b_id):
                        tup = (a_head, a_id, b_head, b_id)
                        if tup not in my_map:
                            my_map[tup] = 0
                        my_map[tup] += 1
                    if (a_head == b_id and a_id != b_head):
                        tup = (b_head, b_id, a_head, a_id)
                        if tup not in my_map:
                            my_map[tup] = 0
                        my_map[tup] += 1
        self.second_order_map = my_map



    def make_wighted_graph(self):
        for sentence in self.senteces:
            s_sentence = self.split_sentence(sentence)
            for line in s_sentence:
                id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel = line
                if head not in self.graph:
                    self.graph[head] = {}
                    self.graph[head][id] = 1
                elif id not in self.graph[head]:
                        self.graph[head][id] = 1
                else:
                    self.graph[head][id] += 1


    def check_sentences(self):
        sentence = self.senteces[0]
        s_sentence = self.split_sentence(sentence)
        words = []
        for id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel in s_sentence:
            words.append(word)
        self.words = words
        for sentence in self.senteces[1:]:
            s_sentence = self.split_sentence(sentence)
            words = []
            for id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel in s_sentence:
                words.append(word)
            if self.words != words:
                print "repeated senteces are'nt the same"
                print words
                print self.words

    def graph_to_inverse_values(self, graph_):
        minus_graph = {}
        for v in graph_:
            minus_graph[v] = {}
            for u in graph_[v]:
                minus_graph[v][u] = -graph_[v][u]
        return minus_graph


    def split_sentence(self, sentence):
        splitted_sentence = []
        for line in sentence:
            s_line = line.split()
            id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel = s_line[:8]
            splitted_sentence.append([id, word, lemma, CPOSTAG, POSTAG, FEATS, head, deprel])
        return splitted_sentence


def file_list_2_wighted_graph(in_list_file, output):
    """
    :param in_list_file: file with repeated sentences (created by k-best or by my script /home/administrator/workspace/k_best/edmonds_mst/edmonds_proj/noise_to_repeated_sentences.py)
    :param output: normal conll file with dependency parsers. (to be written by this very function)
    """
    output_obj = open(output, 'w')
    list_of_repeated_sentences = edmonds_utils.file_2_repeated_sentences(in_list_file)
    for repeated_sentence in list_of_repeated_sentences:
        rs = RepeatedSentence(repeated_sentence)
        rs.check_sentences()
        rs.make_wighted_graph()
        rs.mst_graph_athityakumr()
        rs.mst2conll()
        output_obj.write(rs.conll_sentence_2_str())
        output_obj.write("\n")



def run_mst_on_setups():
    DATA_DIR = "/home/administrator/workspace/k_best/edmonds_mst/data/data_as_rep_sentences"
    RESULTS_DIR = "/home/administrator/workspace/k_best/edmonds_mst/data/results/mst_first_order_results"
    setup_dirs = [os.path.join(DATA_DIR, l) for l in os.listdir(DATA_DIR)]

    for setup_dir in setup_dirs:
        print setup_dir
        for root, dirs, files in os.walk(setup_dir):
            if 'train' in root:
                continue

            new_dir = root.replace(DATA_DIR, RESULTS_DIR)
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
            for file in files:
                file_path = os.path.join(root, file)
                outfile_path = file_path.replace(DATA_DIR, RESULTS_DIR)
                # print file_path
                # print outfile_path


                file_list_2_wighted_graph(file_path, outfile_path)



def get_accuracy(a_file, gold_files_list):
    splitted = a_file.split(os.sep)
    idx = a_file.split(os.sep).index('conll')
    language = splitted[idx+1]
    try:
        gold = [l for l in gold_files_list if language in l][0]
        return language, precision_calc.precision_calc_wrap(gold, a_file)
    except:
        print "Can't get accuracy of {0}".format(language)
        return -1, -1


def print_accuracy(a_file, gold_files_list):
    splitted = a_file.split(os.sep)
    idx = a_file.split(os.sep).index('conll')
    language = splitted[idx+1]
    try:
        gold = [l for l in gold_files_list if language in l][0]
        print language, precision_calc.precision_calc_wrap(gold, a_file)
    except:
        print "Can't get accuracy of {0}".format(language)


def check_results():
    gold_conll = "/home/administrator/workspace/k_best/lianghuang/parserData/sr_data_18_languages/unitag_full_data_no_parentheses/gold_conll"
    gold_files = [os.path.join(gold_conll, l) for l in os.listdir(gold_conll) if 'train' not in l]
    RESULTS_DIR = "/home/administrator/workspace/k_best/edmonds_mst/data/results/mst_first_order_results"
    RESULTS_DIR = "/home/administrator/workspace/k_best/edmonds_mst/data/results/mst_second_order_results/alpha_1_beta_0.5"
    setup_dirs = [os.path.join(RESULTS_DIR, l) for l in os.listdir(RESULTS_DIR)]
    for setup_dir in setup_dirs:
        print "\n"
        print setup_dir
        methods = [os.path.join(setup_dir, l) for l in os.listdir(setup_dir)]
        for method in methods:

            print method.split(os.sep)[-1] + "\n"
            res = {}
            for root, dirs, files in os.walk(method):
                for file_ in files:

                    # print_accuracy(os.path.join(root, file_), gold_files)
                    language, accuracy = get_accuracy(os.path.join(root, file_), gold_files)
                    res[language] = accuracy

            for language in languages:
                if language in res:
                    print language, res[language]
                else:
                    print "{0} not found".format(language)


def main():
    pass


def print_a_graph():
    file = '/home/administrator/workspace/k_best/edmonds_mst/data/data_as_rep_sentences/data_for_reranker_k_minus_1/out_noise_dir/test/conll/arabic/arabic_PADT_test.conll/arabic_PADT_test.conll_18'
    list_of_repeated_sentences = edmonds_utils.file_2_repeated_sentences(file)
    repeated_sentence = list_of_repeated_sentences[0]
    rs = RepeatedSentence(repeated_sentence)
    rs.make_wighted_graph()
    rs.print_graph(rs.graph)

if __name__ == "__main__":
    check_results()
    # print_a_graph()



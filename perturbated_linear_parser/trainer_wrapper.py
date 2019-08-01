import os
import subprocess
import tempfile
import random

from utils import conll_file_2_sentences, sentences_2_conll_file, liang_file_2_sentences, sentences_2_liang_file


python_interpeter_windows = "C:\Users\RYAZDI\AppData\Local\Continuum\Anaconda2\python "
multitrainer = 'python ../liang_parser/multitrainer.py'
parser = "../liang_parser/parser.py"
template_words = "../dp_models/q2.templates"
template_no_words = "../dp_models/templates_no_words"


class TrainRunUnit:
    def __init__(self, language, input_file, model, test_file=None, res_output=None):
        """
        Args:
            language:
            dev_file:
            train_file:
            model: - the output of the train session
            test_file:
            res_output: the output of the test session
        """

        self.language = language
        self.input_file = input_file
        self.dev_file, self.train_file = self.split_data_files(input_file) #TODO: FIX IT
        #self.dev_file, self.train_file = input_file,input_file
        self.model = model
        self.test_file = test_file
        self.output = self.get_output_name(res_output)


    def split_data_files(self, train_file):
        sentences = liang_file_2_sentences(train_file)
        number_of_sentences = len(sentences)
        all_sentences_indexes = range(number_of_sentences)
        sample_10_percent_indexes = random.sample(range(number_of_sentences), number_of_sentences / 10)
        sample_10_percent_indexes = sorted(sample_10_percent_indexes)
        rest_indexes = [l for l in all_sentences_indexes if l not in sample_10_percent_indexes]
        sample_10_percent_sentences = [sentences[l] for l in sample_10_percent_indexes]
        rest_sentences = [sentences[l] for l in rest_indexes]
        train_file = tempfile.NamedTemporaryFile(delete=False)
        sentences_2_liang_file(rest_sentences, train_file.name)
        dev_file = tempfile.NamedTemporaryFile(delete=False)
        sentences_2_liang_file(sample_10_percent_sentences, dev_file.name)
        return dev_file.name, train_file.name

    def __repr__(self):
        str_ = """language: {0}
        input: {6}
        train: {1}
        develop: {2}
        gold: {3}
        model: {4}
        output: {5}
        """.format(self.language,
                   self.train_file,
                   self.dev_file,
                   self.test_file,
                   self.model,
                   self.output,
                   self.input_file)
        return str_

    def get_output_name(self, res_output):
        if not res_output:
            return None

        base = os.path.basename(self.test_file)
        return os.path.join(res_output, base)

    def train_with_words(self):
        run_command = [
            multitrainer, "--train", self.train_file,
            "--dev", self.dev_file,
            "--out", self.model, "-w", template_words
        ]
        print " ".join(run_command)
        p = subprocess.Popen(" ".join(run_command), shell=True)
        p.communicate()

    def train_no_words(self):
        run_command = [
            multitrainer, "--train", self.train_file,
            "--dev", self.dev_file,
            "--out", self.model, "-w", template_no_words
        ]
        #print " ".join(run_command)
        p = subprocess.Popen(" ".join(run_command), shell=True)
        p.communicate()

    def test(self):
        run_command = [
            parser, '-w', self.model, "--input_file", self.test_file, ">", self.output
        ]
        #print " ".join(run_command)
        p = subprocess.Popen(" ".join(run_command), shell=True)
        p.communicate()


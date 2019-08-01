import subprocess
import os

# parser = "../liang_parser/parser.py"

this_file = __file__
my_code = os.path.dirname(this_file)
final_parser = os.path.dirname(my_code)
parser = os.path.join(final_parser,
                      'liang_parser',
                      'parser.py',
                      )

class ParserRunUnit:
    def __init__(self, language, input_file, model, res_output,
                 noise=False, noise_method=None, mu=None, sigma=None, noise_file_path=None):
        """
        Args:
            language:
            input_file: file to parse
            model: trained model
            res_output: the output of the test session
            noise: Boolean
            noise_method: add/multiply
            mu: mean
            sigma: std
            noise_file_path: path to file with vector of noises, different noises for each coefficient
        """
        self.language = language
        self.input_file = input_file
        self.model = model
        self.output = res_output
        self.noise = noise
        self.noise_method = noise_method
        self.mu = mu
        self.sigma = sigma
        self.noise_file_path = noise_file_path

    def __repr__(self):
        str = """line[index]
        language: {0}
        input_file: {1}
        model: {2}
        output: {3}
        noise: {4}
        noise_method: {5}
        mu: {6}
        sigma: {7}
        """.format(self.language,       #1
                   self.input_file,     #2
                   self.model,          #3
                   self.output,         #4
                   self.noise,          #5
                   self.noise_method,   #6
                   self.mu,             #7
                   self.sigma,
                   )
        return str

    def parse_no_words(self,k_best=False):
        run_command = ["python "+parser, '-w', self.model, "--input_file", self.input_file]
        if self.noise:
            run_command += self.add_noise_to_cmd()
        if (k_best):
            run_command += ['-k',str(k_best),'-b 8000']
        run_command += [">>", self.output]
        print " ".join(run_command)
        p = subprocess.Popen(" ".join(run_command), shell=True)
        p.communicate()

    def add_noise_to_cmd(self):
        noise_cmd = ["--noise",
                     "--noise_method", self.noise_method,
                     "--mu", self.mu,
                     "--sigma", self.sigma,
                     ]
        if self.noise_file_path:
            noise_cmd += ["--noise_file_path", self.noise_file_path]
        return noise_cmd




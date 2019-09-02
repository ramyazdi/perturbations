# Perturbations for structured prediction in NLP

The code base for the paper "Perturbation Based Learning for Structured NLP Tasks with Application to Dependency Parsing", Transactions of the Association for Computational Linguistics (TACL), July 2019.\
Please cite this paper if you use our code.


## Prerequisites
The code was implemented in python 2.7 with anaconda environment. 
All requirements are included in the requirements.txt file. They can be installed by running the following command: pip install -r requirements.txt


## Data
All tasks were executed over the [UD V2.0](https://universaldependencies.org) data set, hence, data should be formatted in UD format (CoNLL).\
Data folders should be located at the following hierarchy:\
--perturbations 
  - Data 
    - example_language_1
     - xxx_train.conllu
     - xxx_dev.conllu
     - test
       - xxx_test.conllu (can be more than one file)
    - example_language_2
     - yyy_train.conllu
     - yyy_dev.conllu
     - test
       - yyy_test.conllu (can be more than one file)
   - perturbated_liang_parser
   - 
   - 


## Tasks
### Cross-Lingual Dependency Parsing 
To train a CL linear parsing model execute the following command:

``` 
python perturbated_linear_parser/cross_lingual_exec_process_parallel.py --train_models [ignore if only an inference step is needed] --noise_method ['a' for additive or 'm' for multiplicative noise] 
```

After executing the above command, an evaluation file should be created per each language.

### Mono-Lingual Dependency Parsing 
To train a mono-lingual linear parsing model execute the following command:
``` 
python perturbated_linear_parser/mono_lingual_exec_process_parallel.py --train_models [ignore if only an inference step is needed] --noise_method ['a' for additive or 'm' for multiplicative noise] --eval_method ['oracle' for fully supervised setup or 'unsupervised_min_sim_between_max_sim_to_1best' for lightly supervised setup] 
``` 

### Cross-Lingual POS Tagging
To train a CL neural tagging model execute the following command:
``` 
python neural_pos_tagger/exec_pos_tagger.py --source_language [path/to/source/language/directory] --num_epochs 200 --embedding_path [path/to/fasttext/embeddings/directory] --alignment_path [path/to/alignment/matrices/directory] 
``` 

*)Fasttext's embeddings can be downloaded from [here](https://fasttext.cc/docs/en/crawl-vectors.html) \
**)The aligment matrices can be downloaded from [here](https://github.com/Babylonpartners/fastText_multilingual)


This repository is partially based on the git repository [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2)





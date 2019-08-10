# Perturbations for structured prediction in NLP

The code base for the paper "Perturbation Based Learning for Structured NLP Tasks with Application to Dependency Parsing", Transactions of the Association for Computational Linguistics (TACL), July 2019.
Please cite this paper if you use our code.

## Prerequisites
The code was implemented in python 2.7 with anaconda environment. 
All requirements are included in the requirements.txt file. They can be installed by running the following command: pip install -r requirements.txt


## Data
All tasks were executed over the [UD V2.0](https://universaldependencies.org) data set, hence, data should be formatted in UD format (CoNLL).
Data folders should be located at the following hierarchy:
--perturbations
	--Data
	   --example_lng_1
	      --xxx_train.conllu
	      --xxx_dev.conllu
	      --test
	        --xxx_test.conllu (can be more than one file)
	   --example_lng_2
 	      --yyy_train.conllu
	      --yyy_dev.conllu
	      --test
	        --yyy_test.conllu (can be more than one file)
	--perturbated__liang_parser
	.
	.
	

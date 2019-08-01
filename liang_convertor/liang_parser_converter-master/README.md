A Converter for Liang's Linear-Time Dynamic Programming Parser input/output

Lingpeng Kong (lingpenk at cs dot cmu dot edu)

Aug 31, 2013

Introduction:
 
 This is a simple script which can convert the CONLL format dependency parse tree to the format used in Liang's parser and the other way around. 

Usage: 

	java -cp liangconverter.jar converter.LiangTreeConverter [l2c/c2l] [input_file] [output_file]
		
		[l2c/c21]: 		enter l2c if you are converting liang's format to conll; enter c2l if you are converting from conll to liang's format.
		[input_file]: 	the path for the input file.
		[output_file]: 	the path for the output file.
	
	example:
		java -cp liangconverter.jar converter.LiangTreeConverter l2c 22.dep.4way output.conll
		java -cp liangconverter.jar converter.LiangTreeConverter l2c input.conll output.liang

References:

 1. Liang Huang and Kenji Sagae (2010).
   Dynamic Programming for Linear-Time Incremental Parsing.
   Proceedings of ACL.

 2. Liang Huang, Suphan Fayong, and Yang Guo (2012).
   Structured Perceptron with Inexact Search.
   Proceedings of NAACL.

Permission is granted for anyone to copy, use, or modify these programs and
accompanying documents for purposes of research or education, provided this
copyright notice is retained, and note is made of any changes that have been
made.

These programs and documents are distributed without any warranty, express or
implied.  As the programs were written for research purposes only, they have
not been tested to the degree that would be advisable in any important
application.  All use of these programs is entirely at the user's own risk.
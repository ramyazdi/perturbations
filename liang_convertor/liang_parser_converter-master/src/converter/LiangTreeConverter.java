package converter;

import java.io.File;
import java.util.ArrayList;

import util.LineReader;
import util.LineWriter;

/**
 * LiangTreeConverter provides functions to convert a Conll file to the format
 * used by Liang's parser and from format used by Liang's parser to Conll
 * format.
 * 
 * @author Lingpeng Kong (lingpenk at cs.cmu.edu)
 * @date Aug 31, 2013
 * 
 */
public class LiangTreeConverter {

	public static void main(String[] args) {
		File inputfile;
		File outputfile;
		String function;

		if (args.length != 3) {
			System.out
					.println("Usage: java -cp liangconverter.jar converter.LiangTreeConverter [l2c/c2l] [input_file] [output_file]\n[l2c/c21]: enter l2c if you are converting liang's format to conll; enter c2l if you are converting from conll to liang's format.\n[input_file]: the path for the input file.\n[output_file]: the path for the output file.\n");
			return;
		} else {
			inputfile = new File(args[1]);
			outputfile = new File(args[2]);
			function = args[0];
		}

		LineReader lr = new LineReader(inputfile);
		LineWriter lw = new LineWriter(outputfile);

		if (function.equals("l2c")) {
			while (lr.hasNextLine()) {
				String line = lr.readNextLine().trim();
				if (line.equals(""))
					continue;
				// System.out.println(line);
				LiangTreeNode ltn = LiangTreeNode.parse(line);
				// System.out.println(ltn.getLiangFormatString());
				ConllStru cs = new ConllStru(ltn);
				lw.writeln(cs.getPrintString());
			}
		} else if (function.equals("c2l")) {

			ArrayList<String> list = new ArrayList<String>();
			while (lr.hasNextLine()) {
				String line = lr.readNextLine();
				if (line.equals("")) {
					ConllStru cs = new ConllStru(list);
					lw.writeln(cs.toLiangTree().getLiangFormatString());
					list = new ArrayList<String>();
					continue;
				}
				list.add(line);
			}

		}
		lw.closeAll();
		lr.closeAll();

	}

}

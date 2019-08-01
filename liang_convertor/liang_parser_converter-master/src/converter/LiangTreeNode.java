package converter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import util.LineReader;

/**
 * LiangTree represents a node in Liang's Format, which contains the word (and the
 * POS). Of course, it can be the head node.
 * 
 * @author Lingpeng Kong (lingpenk at cs.cmu.edu)
 * @date Aug 31, 2013
 * 
 */
public class LiangTreeNode {
	private String content;
	private int index;
	private List<LiangTreeNode> leftChildren;
	private List<LiangTreeNode> rightChildren;
	
	private final static String regexIndexLabel = "@_L\\d+_@";
	
	public static LiangTreeNode parse(String s){
		HashMap<String, Integer> indexMap = new HashMap<String, Integer>();
		
		s = s.trim();
		String sf = "";
		int i = 0;
		while(i < s.length()){
			char c = s.charAt(i);
			if(c == '(' || c == ')' || c ==' '){
				sf = sf + c; // copy the char into the new string
				i++;
			}else{
				/* meet a token here, read it */
				String token = "" + c;
				i++;
				while(i < s.length()){
					char cc = s.charAt(i);
					if(cc == '(' || cc == ')' || cc ==' '){
						break;
					}else{
						token = token + cc;
						i++;
					}
				}
				//System.out.println(token);
				if(!indexMap.containsKey(token)){
					indexMap.put(token, 0);
				}else{
					indexMap.put(token, indexMap.get(token) + 1);
					token = token + "@_L"+ indexMap.get(token) +"_@";
				}
				sf = sf + token;
				
			}
		}
		//System.out.println(sf);
		//System.out.println(sf.replaceAll(regexIndexLabel, ""));
		
		indexMap.clear();
		String[] terms = sf.replaceAll("\\(", "").replaceAll("\\)", "").split(" +");
		for(int j = 0; j < terms.length; j++){
			//System.out.println(terms[j]);
			indexMap.put(terms[j], (j+1));
		}
		
		return parse(sf, indexMap);
		
	}
	
	private static LiangTreeNode parse(String sf, HashMap<String, Integer> indexMap){
		sf = sf.trim();
		if(sf.startsWith("(") && sf.endsWith(")")){
			sf = sf.substring(1, sf.length()-1).trim(); // delete the brackets
			int i = 0;
			LiangTreeNode ltn = new LiangTreeNode();
			boolean beforeHead = true;
			while(i < sf.length()){
				char c = sf.charAt(i);
				if(c == ' '){
					i++;
					continue;
				}
				if(c == '('){
					/* a new bracket structure should be read in here */
					int leftBracketMore = 1;
					String term = "" + c;
					i++;
					while(i < sf.length()){
						char cc = sf.charAt(i);
						if(cc == ')'){
							leftBracketMore = leftBracketMore - 1;
						}else if(cc == '('){
							leftBracketMore = leftBracketMore + 1;
						}
						term = term + cc;
						i++;
						if(leftBracketMore == 0){
							if(beforeHead){
								ltn.leftChildren.add(parse(term,indexMap));
							}else{
								ltn.rightChildren.add(parse(term,indexMap));
							}
							break;
						}
					}
				}else{
					/* here we start to read the head node for this stage */
					String term = "" + c;
					i++;
					while(i < sf.length()){
						char cc = sf.charAt(i);
						if(cc == ' '){
							break;
						}else{
							term = term + cc;
							i++;
						}
					}
				    ltn.setContent(term.replaceAll(regexIndexLabel, ""));
				    ltn.setIndex(indexMap.get(term));
				    beforeHead = false;
				}
			}
			return ltn;
		}
		LiangTreeNode ltn = new LiangTreeNode();
		ltn.setContent(sf.replaceAll(regexIndexLabel, ""));
	    ltn.setIndex(indexMap.get(sf));
	    return ltn;
	}
	
	public static void main(String[] args){
		LineReader lr = new LineReader("Tree");
		while(lr.hasNextLine()){
			String line = lr.readNextLine().trim();
			if(line.equals("")) continue;
			//System.out.println(line);
			LiangTreeNode ltn = parse(line);
			//System.out.println(ltn.getLiangFormatString());
			if(!ltn.getLiangFormatString().equals(line)){
				System.err.println("alert");
				System.out.println(line);
				System.out.println(ltn.getLiangFormatString());
			}
			ConllStru cs = new ConllStru(ltn);
			System.out.println(cs.getPrintString());
		}
		
		lr.closeAll();
	}

	
	public LiangTreeNode() {
		leftChildren = new ArrayList<LiangTreeNode>();
		;
		rightChildren = new ArrayList<LiangTreeNode>();
	}
	
	public String getContent() {
		return content;
	}

	public void setContent(String content) {
		this.content = content;
	}

	public int getIndex() {
		return index;
	}

	public void setIndex(int index) {
		this.index = index;
	}

	public List<LiangTreeNode> getLeftChildren() {
		return leftChildren;
	}

	public void setLeftChildren(List<LiangTreeNode> leftChildren) {
		this.leftChildren = leftChildren;
	}

	public List<LiangTreeNode> getRightChildren() {
		return rightChildren;
	}

	public void setRightChildren(List<LiangTreeNode> rightChildren) {
		this.rightChildren = rightChildren;
	}

	public String getLiangFormatString() {
		if (leftChildren.size() == 0 && rightChildren.size() == 0) {
			return "(" + content + ")";
		} else {
			String output = "";
			for (LiangTreeNode ltn : leftChildren) {
				output = output + ltn.getLiangFormatString() + " ";
			}
			output = output + content;
			for (LiangTreeNode ltn : rightChildren) {
				output = output + " " + ltn.getLiangFormatString();
			}
			output = "(" + output + ")";
			return output;
		}
	}

}

package converter;
/**
 * A ConllNode is just a collection of information about a word in the sentence.
 * 
 * @author Lingpeng Kong (lingpenk at cs.cmu.edu)
 * @date Aug 31, 2013
 * 
 */
public class ConllNode implements Comparable<ConllNode>{

	private String POS;
	private String word;
	private int index;
	private int parentIndex;

	public ConllNode(int index, String word, String POS, int parentIndex) {
		super();
		this.POS = POS;
		this.word = word;
		this.index = index;
		this.parentIndex = parentIndex;
	}
	
	public String getPOS() {
		return POS;
	}

	public void setPOS(String pOS) {
		POS = pOS;
	}

	public String getWord() {
		return word;
	}

	public void setWord(String word) {
		this.word = word;
	}

	public int getIndex() {
		return index;
	}

	public void setIndex(int index) {
		this.index = index;
	}

	public int getParentIndex() {
		return parentIndex;
	}

	public void setParentIndex(int parentIndex) {
		this.parentIndex = parentIndex;
	}
	
	public String getContent(boolean includePOS){
		String output = new String(word);
		if(includePOS){
			output = output + "/" + POS;
		}
		return output;
	}

	public int compareTo(ConllNode arg0) {
		return index - arg0.index;
	}

}

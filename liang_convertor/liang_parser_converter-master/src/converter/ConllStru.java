package converter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import util.LineReader;

/**
 * ConllStru represents the essential content for a sentence in Conll Format,
 * which for the Liang's parser is (Index, Word, POS(optional) and Parent Node).
 * 
 * @author Lingpeng Kong (lingpenk at cs.cmu.edu)
 * @date Aug 31, 2013
 * 
 */
public class ConllStru {
	private static boolean includePOS = true;
	private ConllNode[] nodeList;
	/* construct a ConllStru from a Liang Tree */
	public ConllStru(LiangTreeNode tree) {
		ArrayList<ConllNode> conllNodes = new ArrayList<ConllNode>();
		parse(tree, conllNodes, 0);
		Collections.sort(conllNodes);
		
		nodeList = new ConllNode[conllNodes.size()];
		for(int i = 0; i < nodeList.length; i++){
			int index = conllNodes.get(i).getIndex();
			int parentIndex =  conllNodes.get(i).getParentIndex();
			String word =  conllNodes.get(i).getWord();
			String POS = conllNodes.get(i).getPOS();
			nodeList[i] = new ConllNode(index, word, POS, parentIndex);
		}
		
	}
	
	private void parse(LiangTreeNode treeNode, ArrayList<ConllNode> conllNodes, int parentIndex){
		String word = "";
		String POS = "";
		String cont = treeNode.getContent().trim();
		int t = cont.lastIndexOf("/");
		if(t > 0){
			word = cont.substring(0, t);
			POS = cont.substring(t+1, cont.length());
		}else{
			word = cont;
		}
		
		ConllNode cn = new ConllNode(treeNode.getIndex(), word, POS, parentIndex);
		conllNodes.add(cn);
		if(treeNode.getLeftChildren().size()>0){
			for(LiangTreeNode ltn : treeNode.getLeftChildren()){
				parse(ltn, conllNodes, treeNode.getIndex());
			}
		}
		if(treeNode.getRightChildren().size()>0){
			for(LiangTreeNode ltn : treeNode.getRightChildren()){
				parse(ltn, conllNodes, treeNode.getIndex());
			}
		}
	}
	
	public ConllStru(List<String> lines) {
		nodeList = new ConllNode[lines.size()];
		for(int i = 0; i < nodeList.length; i++){
			String[] args = lines.get(i).split("\\t");
			int index = Integer.parseInt(args[0]);
			int parentIndex = Integer.parseInt(args[6]);
			String word = args[1];
			String POS = args[4];
			nodeList[i] = new ConllNode(index, word, POS, parentIndex);
		}
		
		
	}
	
	public LiangTreeNode toLiangTree(){
		LiangTreeNode ltn = new LiangTreeNode();
		for(int j = 0; j < nodeList.length; j++){
			ConllNode cnt = nodeList[j];
			if(cnt.getParentIndex() == 0){
				ltn.setContent(cnt.getContent(includePOS));
				ltn.setIndex(cnt.getIndex());
			}
		}
		growLiangTreeNode(ltn);
		//System.out.println(ltn.getLiangFormatString());
		return ltn;
	}
	
	private void growLiangTreeNode(LiangTreeNode ltn){
		for(int j = 0; j < nodeList.length; j++){
			ConllNode cnt = nodeList[j];
			if(cnt.getParentIndex() == ltn.getIndex()){
				LiangTreeNode ltnt = new LiangTreeNode();
				ltnt.setIndex(cnt.getIndex());
				ltnt.setContent(cnt.getContent(includePOS));
				if(cnt.getIndex() < ltn.getIndex()){
					ltn.getLeftChildren().add(ltnt);
				}else{
					ltn.getRightChildren().add(ltnt);
				}
				growLiangTreeNode(ltnt);
			}
		}
	}

	
	public ConllNode[] getNodeList() {
		return nodeList;
	}

	public void setNodeList(ConllNode[] nodeList) {
		this.nodeList = nodeList;
	}
	
	public String getPrintString(){
		String output = "";
		for (int i = 0; i < nodeList.length; i++) {
			// 4 was _ VB VBD _ 7 cop _ _
			output = output + nodeList[i].getIndex() + "\t" + nodeList[i].getWord() + "\t_\t"
					+ nodeList[i].getPOS() + "\t" + nodeList[i].getPOS()
					+ "\t_\t" + nodeList[i].getParentIndex() + "\t_\t_\t_\n";
		}
		return output;
	}
	
	
	public static void main(String[] args) {
		LineReader lr = new LineReader("TestTree");
		ArrayList<String> list = new ArrayList<String>();
		while(lr.hasNextLine()){
			String line = lr.readNextLine();
			if(line.equals("")){
				ConllStru cs = new ConllStru(list);
				cs.getPrintString();
				list = new ArrayList<String>();
				continue;
			}
			list.add(line);
		}
		lr.closeAll();
		
	}

}

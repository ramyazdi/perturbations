package util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * This is a class which make the writer in java more convenient to use
 * Dec.2, 2012, at Carnegie Mellon
 * @author Lingpeng Kong
 */
public class LineWriter {
  private File outFile;
  private FileWriter fw;
  private BufferedWriter bw;
  
  public LineWriter(File out){
    outFile = out;
    initWriter();
  }
  
  public LineWriter(String outAdd){
    outFile = new File(outAdd);
    initWriter();
  }
  public void writeln(){
    try {
      bw.write("\n");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  public void writeln(String s){
    try {
      bw.write(s+"\n");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  public void write(String s){
    try {
      bw.write(s);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  public void closeAll(){
    try {
      bw.close();
      fw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  private void initWriter(){
    try {
      fw = new FileWriter(outFile);
      bw = new BufferedWriter(fw);
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
  
  /**
   * @param args
   */
  public static void main(String[] args) {

  }

}

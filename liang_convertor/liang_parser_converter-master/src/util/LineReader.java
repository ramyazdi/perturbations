package util;



import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * This is a class that can read file line by line and split them into array of Strings at the same
 * time
 * 
 * @author Lingpeng Kong
 * 
 */
public class LineReader {
  private File readFile;

  private FileReader fr;

  private BufferedReader br;

  public LineReader(String fileDir) {
    readFile = new File(fileDir);
    initReaderBuffer();
  }

  public LineReader(File file) {
    readFile = file;
    initReaderBuffer();
  }

  public boolean initReaderBuffer() {
    try {
      fr = new FileReader(readFile);
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      return false;
    }
    br = new BufferedReader(fr);
    return true;
  }

  public boolean hasNextLine() {
    try {
      if (br.ready()) {
        return true;
      }
    } catch (IOException e) {
      e.printStackTrace();
      return false;
    }
    return false;
  }

  public String readNextLine() {
    if (hasNextLine()) {
      try {
        return br.readLine();
      } catch (IOException e) {
        e.printStackTrace();
        return null;
      }
    } else {
      return null;
    }
  }

  public String[] readAndSplit(String splitRex) {
    String line = readNextLine();
    if (line != null) {
      return line.split(splitRex);
    } else {
      return null;
    }
  }

  public void closeAll() {
    try {
      br.close();
      fr.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  /**
   * @param args
   */
  public static void main(String[] args) {
    /*
    LineReader lr = new LineReader(PhraseFinder.inputUnigramDir);
    
    while (lr.hasNextLine()) {
      String[] s = lr.readAndSplit("\\t");
      for (String m : s) {
        System.out.println(m);
      }
      System.out.println();
    }
    lr.closeAll();
    */
    
  }
  

}

package ceka.IWBVT;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.bayes.NaiveBayes;


public class Test_BVT { 
	public static void main(String [] args) {
		BVDcrossvalx bvd = new BVDcrossvalx();

		String dataPath= "./test/ceka/IWBVT/real-world/ablation_BVT/leaves";

		NaiveBayes m_Classifier = new NaiveBayes();

		List files = getFileList(dataPath);

		bvd.setNumFolds(3);
		bvd.setTrainIterations(10);
		
		// 运行基础算法
		bvd.setClassifier(m_Classifier);
		for(int i=0; i<files.size(); i++)
		{
			bvd.setDataFileName(dataPath +"\\"+ files.get(i));
			System.out.print(files.get(i) + ",");
			try {
				bvd.decompose();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	// 获取路径
	public static List getFileList(String path) {
		List list = new ArrayList();
		try {
			File file = new File(path);
			String[] filelist = file.list();
			for (int i = 0; i < filelist.length; i++) {
				if(!filelist[i].equals("Readme.txt")) {
					list.add(filelist[i]);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return list;
	}
}

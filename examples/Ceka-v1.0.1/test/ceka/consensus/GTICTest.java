package ceka.consensus;

import java.io.File;
import java.util.ArrayList;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import ceka.consensus.gtic.GTIC;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.utils.PerformanceStatistic;

public class GTICTest {

	private static Logger log = Logger.getLogger(GTICTest.class);
	
	private static String dataDir = "D:/CekaSpace/Ceka/data/real-world/Processed/GTIC/";
	private static String responseFix = ".response.txt";
	private static String goldFix = ".gold.txt";
	private static String runDir = "D:/CekaSpace/Temp/GTIC/";
	
	private static class Result
	{
		long estimatedTime;
		double acc;
		double auc;
		@SuppressWarnings("unused")
		double aucMax;
		ArrayList<Double> cateAcc = new ArrayList<Double>();
		int [] initialCentroids = null;
	}
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {

		PropertyConfigurator.configure("D:/CekaSpace/Ceka/lib/log4j.properties");
		
		// ten real-word data sets
		
		//String dataName = "trec2010";
		//String dataName = "adult2";
		//String dataName = "valence5";
		//String dataName = "aircrowd6";
		//String dataName = "valence7";
		//String dataName = "leaves16";
		//String dataName = "fej2013";
		//String dataName = "saj2013";
		//String dataName = "leaves9";
		String dataName = "aircrowd11";
		
		File testDir  = new File(runDir);
		if (!testDir.exists())
			testDir.mkdirs();
		
		// load data sets
		String responsePath = dataDir + dataName + responseFix;
		String goldPath = dataDir + dataName + goldFix;
		Dataset dataset = null;
		dataset = FileLoader.loadFile(responsePath, goldPath);
		
		// create GTIC algorithm
		GTIC gtic = new  GTIC( runDir);
		// run algorithm
		gtic.doInference(dataset);
		
		// get results
		Result rst = new Result();	
		rst.estimatedTime = gtic.getExcuteTime();
		rst.initialCentroids = gtic.getInitialCentroids();
		PerformanceStatistic reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		rst.acc = reporter.getAccuracy();
		rst.auc = reporter.getAUC();
		rst.aucMax = reporter.getAUCConvex();
		
		// print results
		for (int k = 0; k < dataset.getCategorySize(); k++)
			rst.cateAcc.add(new Double(reporter.getAccuracyCategory(k)));
		log.info("------------RESULT-----------------------------");
		for (int k = 0; k < dataset.getCategorySize(); k++)
			log.info("Class " + k + " :" +  rst.cateAcc.get(k));
		log.info("Overall Accuracy: " + rst.acc);
		log.info("M-AUC: " +  rst.auc);
		log.info("Running Time: " + rst.estimatedTime);
		//log.info("mAUCMax: " + rst.aucMax);
		String initialCStr = "Initial Centroids: ";
		for (int k = 0; k < rst.initialCentroids.length; k++)
			initialCStr += (rst.initialCentroids[k] + "    ");
		log.info(initialCStr);
	}
}

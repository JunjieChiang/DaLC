/**
 * 
 */
package ceka.consensus;

import java.io.File;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import ceka.consensus.plat.PLAT;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.utils.PerformanceStatistic;

/**
 * @author Zhang
 *
 */
public class WeightedVoteTest {
	
	private static Logger log = Logger.getLogger(WeightedVoteTest.class);
	
	private static String dataDir = "D:/CekaSpace/Ceka/data/real-world/Processed/BinaryBiased/";
	private static String responseFix = ".response.txt";
	private static String goldFix = ".gold.txt";
	private static String runDir = "D:/CekaSpace/Temp/PLAT/";
	
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		PropertyConfigurator.configure("D:/CekaSpace/Ceka/lib/log4j.properties");
		
		//String dataName = "anger.median";
		//String dataName = "disgust.median";
		//String dataName = "fear.median";
		//String dataName = "joy.median";
		//String dataName = "sadness.median";
		//String dataName = "surprise.median";
		//String dataName = "valence.median";
		//String dataName = "anger.minentropy";
		//String dataName = "disgust.minentropy";
		//String dataName = "fear.minentropy";
		//String dataName = "joy.minentropy";
		//String dataName = "sadness.minentropy";
		//String dataName = "surprise.minentropy";
		//String dataName = "valence.minentropy";
		//String dataName = "spam";
		
		String dataName = "duck";
		
		File testDir  = new File(runDir);
		if (!testDir.exists())
			testDir.mkdirs();
		
		// load data sets
		String responsePath = dataDir + dataName + responseFix;
		String goldPath = dataDir + dataName + goldFix;
		Dataset dataset = null;
		dataset = FileLoader.loadFile(responsePath, goldPath);
		
		MajorityVote mv = new MajorityVote();
		mv.doInference(dataset);
		PerformanceStatistic reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		
		log.info("MV accuracy: " + reporter.getAccuracy() + " Roc Area: " + reporter.getAUC() + " Recall: " + reporter.getRecallBinary() + " Precision: " + reporter.getPresicionBinary()
				+ " F1:" +  reporter.getF1MeasureBinary());
		
		PLAT plat = new PLAT();
		plat.doInference(dataset);
		
		reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		
		log.info("PLAT accuracy: " + reporter.getAccuracy() + " Roc Area: " + reporter.getAUC() + " Recall: " + reporter.getRecallBinary() + " Precision: " + reporter.getPresicionBinary()
				+ " F1:" +  reporter.getF1MeasureBinary());
		
		AdaptiveWeightedMajorityVote wv = new AdaptiveWeightedMajorityVote();
		wv.doInference(dataset);
		reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		
		log.info("WMV accuracy: " + reporter.getAccuracy() + " Roc Area: " + reporter.getAUC() + " Recall: " + reporter.getRecallBinary() + " Precision: " + reporter.getPresicionBinary()
				+ " F1:" +  reporter.getF1MeasureBinary());
	}

}

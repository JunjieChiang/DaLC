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
public class PLATTest {

	private static Logger log = Logger.getLogger(GTICTest.class);
	
	private static String dataDir = "D:/CekaSpace/Ceka/data/real-world/Processed/BinaryBiased/";
	private static String responseFix = ".response.txt";
	private static String goldFix = ".gold.txt";
	private static String runDir = "D:/CekaSpace/Temp/PLAT/";
	
	public static void main(String[] args) throws Exception {

		PropertyConfigurator.configure("D:/CekaSpace/Ceka/lib/log4j.properties");
		
		String dataName = "trec2010";
		
		File testDir  = new File(runDir);
		if (!testDir.exists())
			testDir.mkdirs();
		
		// load data sets
		String responsePath = dataDir + dataName + responseFix;
		String goldPath = dataDir + dataName + goldFix;
		Dataset dataset = null;
		dataset = FileLoader.loadFile(responsePath, goldPath);
		
		PLAT plat  = new PLAT();
		plat.setUseQuadraticFitting(true);
		plat.doInference(dataset);
		
		PerformanceStatistic reporter = new PerformanceStatistic();
		reporter.stat(dataset);
		
		log.info("PLAT accuracy: " + reporter.getAccuracy() + " Roc Area: " + reporter.getAUC() + " Recall: " + reporter.getRecallBinary() 
				+ " F1:" +  reporter.getF1MeasureBinary());
	}
}

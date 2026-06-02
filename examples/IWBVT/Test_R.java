package ceka.IWBVT;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import ceka.AVNC.*;
import ceka.IWMV.IWMV;
import ceka.LAWMV.LAWMV;
import ceka.MNLDP.MNLDP;
import ceka.MVNC.MVNC;
import ceka.NWVNC.NWVNC;
import ceka.consensus.MajorityVote;
import ceka.converters.FileLoader;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.utils.Stochastics;
import myqpmatlab.MyQP;

public class Test_T {
	
	
	private static String dataSetArffDir = "./test/ceka/IWBVT/real-world/";

	private static String[] dataSetArddFix = {"leaves"};

	private static String runDir = "./test/ceka/IWBVT/IWBVT-results/";

	/** the number of instances. */
	private static int num_data = dataSetArddFix.length;
	
	private static int num_Folds = 1;
	
	private static Classifier m_classifier = new NaiveBayes();

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		File testDir = new File(runDir);
		if (!testDir.exists())
			testDir.mkdirs();

		// format data
		DecimalFormat df = new DecimalFormat("#.00");

		// write the result to a file
		File f = new File(runDir + "leaves.txt");
		FileWriter fw = new FileWriter(f);
			
		fw.write("dataset\t" + "MV\t" + "MV+\t" + "IWMV\t" + "IWMV+\t" + "LAWMV\t" + "LAWMV+\t" + "MNLDP\t" + "MNLDP+\t" + "AVNC\t" + "AVNC+\t" + "MVNC\t" + "MVNC+\t" + "NWVNC\t" + "NWVNC+\t");
		fw.write("\r\n");
		fw.flush();

		double all_mvf1 = 0.0;
		double all_mvf2 = 0.0;
		double all_mvf3 = 0.0;
		double all_mvf4 = 0.0;
		double all_mvf5 = 0.0;
		double all_mvf6 = 0.0;
		double all_mvf7 = 0.0;
		double all_mvf8 = 0.0;
		double all_mvf9 = 0.0;
		double all_mvf10 = 0.0;
		double all_mvf11 = 0.0;
		double all_mvf12 = 0.0;
		double all_mvf13 = 0.0;
		double all_mvf14 = 0.0;

		for (int data = 0; data < num_data; data++) {
							
			String arffxPath = dataSetArffDir + dataSetArddFix[data] + ".arffx";
			String responsePath=dataSetArffDir + dataSetArddFix[data] + ".response.txt";
			String goldPath=dataSetArffDir + dataSetArddFix[data] + ".gold.txt";		
			Dataset dataset = FileLoader.loadFileX(responsePath, goldPath, arffxPath);
			
			System.out.println(dataSetArddFix[data]);
			
			double result_mvf1 = 0.0;
			double result_mvf2 = 0.0;
			double result_mvf3 = 0.0;
			double result_mvf4 = 0.0;
			double result_mvf5 = 0.0;
			double result_mvf6 = 0.0;
			double result_mvf7 = 0.0;
			double result_mvf8 = 0.0;
			double result_mvf9 = 0.0;
			double result_mvf10 = 0.0;
			double result_mvf11 = 0.0;
			double result_mvf12 = 0.0;
			double result_mvf13 = 0.0;
			double result_mvf14 = 0.0;

			for (int k = 0; k < num_Folds; k++) {
				// 保存一下数据集用来测试消融
//				DataSink.write(runDir + "dataset.arff", dataset);
				
				// MV
				Dataset temp_train_mv = copyDataset(dataset);
				MajorityVote mvf1 = new MajorityVote();
				mvf1.doInference(temp_train_mv);
				
				// 保存一下数据集用来测试消融
//				DataSink.write(runDir + "MV.arff", temp_train_mv);

				// IWMV
				Dataset temp_train_iwmv = copyDataset(temp_train_mv);
				IWMV mvf2 = new IWMV();
				mvf2.doInference(temp_train_iwmv);
				
				// LAWMV
				Dataset temp_train_lawmv = copyDataset(temp_train_mv);
				LAWMV mvf3 = new LAWMV();
				mvf3.doInference(temp_train_lawmv, (int)(0.5*temp_train_lawmv.numInstances() / temp_train_lawmv.numClasses()));

				// MNLDP
				Dataset temp_train_mnldp = copyDataset(temp_train_mv);
				MyQP t1 = new MyQP();
				MNLDP mvf4 = new MNLDP();
				mvf4.setMyQP(t1);
				mvf4.doInference(temp_train_mnldp);
				t1.dispose();
	
				// NWVNC
				Dataset temp_train_nwvnc = copyDataset(temp_train_mv);
				NWVNC newnc = new NWVNC();
				temp_train_nwvnc = newnc.nwvnc(temp_train_nwvnc);
				
				// MVNC
				Dataset temp_train_mvnc = copyDataset(temp_train_mv);
				MVNC mvnc = new MVNC();
				temp_train_mvnc = mvnc.doInference(temp_train_mvnc);
				
				// AVNC
				Dataset temp_train_avnc = copyDataset(temp_train_mv);
				WorkerStat workerStat = new WorkerStat();
				double estimatedMeanProb = workerStat.calculateEstimatedMeanAcc(temp_train_mv);
				double integratedCorrectProb = Stochastics.binomialIntegration(9, estimatedMeanProb);
				int nfold = 10;
				int nModel = 5;
				AdaptiveClassificationFilter acf = new AdaptiveClassificationFilter(nfold,nModel);
				acf.setMinEstimatedNoiseProportion(1-integratedCorrectProb);
				acf.setMaxEstimatedNoiseProportion(1-estimatedMeanProb);
				Classifier[] classifiers4 = new Classifier[5];
				for(int kk=0; kk<5; kk++)
					classifiers4[kk] = new J48();
				acf.filterNoise(temp_train_avnc, classifiers4);
				Dataset cleanSet2=acf.getCleansedDataset();
				Dataset noiseSet2=acf.getNoiseDataset();
				Dataset[] highDatasets=acf.getHighQualityDatasets();
				
				VoteCorrection corrector=new VoteCorrection();
				corrector.correct(noiseSet2, highDatasets, classifiers4, (int)(highDatasets.length*0.5));
				for(int kk=0;kk<noiseSet2.getExampleSize();kk++)
					cleanSet2.addExample(noiseSet2.getExampleByIndex(kk));
				System.out.println("AVNC Completed!");
				
				// 设置分类器
				Classifier m_classifier1 = new NaiveBayes();
				IWBVT m_classifier2 = new IWBVT();
				m_classifier2.setClassifier(m_classifier1);
				
				// MV
				result_mvf1 += CekaUtils.classificationAccuracy(temp_train_mv, 10, 10, m_classifier);
				result_mvf2 += CekaUtils.classificationAccuracyforOneStage(temp_train_mv, 10, 10, m_classifier2);
				
				// IWMV
				result_mvf3 += CekaUtils.classificationAccuracy(temp_train_iwmv, 10, 10, m_classifier);
				result_mvf4 += CekaUtils.classificationAccuracyforOneStage(temp_train_iwmv, 10, 10, m_classifier2);
				
				// LAWMV
				result_mvf5 += CekaUtils.classificationAccuracy(temp_train_lawmv, 10, 10, m_classifier);
				result_mvf6 += CekaUtils.classificationAccuracyforOneStage(temp_train_lawmv, 10, 10, m_classifier2);
				
				// MNLDP
				result_mvf7 += CekaUtils.classificationAccuracy(temp_train_mnldp, 10, 10, m_classifier);
				result_mvf8 += CekaUtils.classificationAccuracyforOneStage(temp_train_mnldp, 10, 10, m_classifier2);
				
				// AVNC
				result_mvf9 += CekaUtils.classificationAccuracy(cleanSet2, 10, 10, m_classifier);
				result_mvf10 += CekaUtils.classificationAccuracyforOneStage(cleanSet2, 10, 10, m_classifier2);
				
				// MVNC
				result_mvf11 += CekaUtils.classificationAccuracy(temp_train_mvnc, 10, 10, m_classifier);
				result_mvf12 += CekaUtils.classificationAccuracyforOneStage(temp_train_mvnc, 10, 10, m_classifier2);
				
				// NWVNC
				result_mvf13 += CekaUtils.classificationAccuracy(temp_train_nwvnc, 10, 10, m_classifier);
				result_mvf14 += CekaUtils.classificationAccuracyforOneStage(temp_train_nwvnc, 10, 10, m_classifier2);
			}

			String dataName = dataSetArddFix[data].split("\\.")[0];
			
			fw.write(dataName + "\t"
					+ df.format(result_mvf1 / num_Folds) + "\t"
					+ df.format(result_mvf2 / num_Folds) + "\t"
					+ df.format(result_mvf3 / num_Folds) + "\t"
					+ df.format(result_mvf4 / num_Folds) + "\t"
					+ df.format(result_mvf5 / num_Folds) + "\t"
					+ df.format(result_mvf6 / num_Folds) + "\t"
					+ df.format(result_mvf7 / num_Folds) + "\t"
					+ df.format(result_mvf8 / num_Folds) + "\t"
					+ df.format(result_mvf9 / num_Folds) + "\t"
					+ df.format(result_mvf10 / num_Folds) + "\t"
					+ df.format(result_mvf11 / num_Folds) + "\t"
					+ df.format(result_mvf12 / num_Folds) + "\t"
					+ df.format(result_mvf13 / num_Folds) + "\t"
					+ df.format(result_mvf14 / num_Folds) + "\t");

			fw.write("\r\n");
			fw.flush();
			
			all_mvf1 += result_mvf1 / num_Folds;
			all_mvf2 += result_mvf2 / num_Folds;
			all_mvf3 += result_mvf3 / num_Folds;
			all_mvf4 += result_mvf4 / num_Folds;
			all_mvf5 += result_mvf5 / num_Folds;
			all_mvf6 += result_mvf6 / num_Folds;	
			all_mvf7 += result_mvf7 / num_Folds;
			all_mvf8 += result_mvf8 / num_Folds;
			all_mvf9 += result_mvf9 / num_Folds;	
			all_mvf10 += result_mvf10 / num_Folds;	
			all_mvf11 += result_mvf11 / num_Folds;	
			all_mvf12 += result_mvf12 / num_Folds;	
			all_mvf13 += result_mvf13 / num_Folds;	
			all_mvf14 += result_mvf14 / num_Folds;	
		}
		fw.write("average " + "\t" 
				+ df.format(all_mvf1 / num_data) + "\t"
				+ df.format(all_mvf2 / num_data) + "\t"
				+ df.format(all_mvf3 / num_data) + "\t"
				+ df.format(all_mvf4 / num_data) + "\t"
				+ df.format(all_mvf5 / num_data) + "\t"
				+ df.format(all_mvf6 / num_data) + "\t"
				+ df.format(all_mvf7 / num_data) + "\t"
				+ df.format(all_mvf8 / num_data) + "\t"
				+ df.format(all_mvf9 / num_data) + "\t"
				+ df.format(all_mvf10 / num_data) + "\t"
				+ df.format(all_mvf11 / num_data) + "\t"
				+ df.format(all_mvf12 / num_data) + "\t"
				+ df.format(all_mvf13 / num_data) + "\t"
				+ df.format(all_mvf14 / num_data) + "\t");

		fw.write("\r\n");
		fw.flush();
		fw.close();
	}

	public static double randdouble(double max, double min) {
		return (Math.random() * (max - min) + min);
	}

	public static Dataset copyDataset(Dataset dataset) {
		Dataset copyDataset = new Dataset(dataset, 0);
		for (int k = 0; k < dataset.getExampleSize(); k++) {
			Example example = dataset.getExampleByIndex(k);
			copyDataset.addExample(example);
		}
		for (int k = 0; k < dataset.getCategorySize(); k++) {
			Category category = dataset.getCategory(k);
			copyDataset.addCategory(category);
		}
		for (int k = 0; k < dataset.getWorkerSize(); k++) {
			Worker worker = dataset.getWorkerByIndex(k);
			copyDataset.addWorker(worker);
		}
		return copyDataset;
	}
	
	//将instances封装成dataset
	public static Dataset InstancesToDataset(Instances instances,Dataset dataset1) {
		Dataset dataset = new Dataset(instances,instances.numInstances());
		for(int m = 0;m < dataset1.getCategorySize();m++) {
			Category cate = dataset1.getCategory(m);
			dataset.addCategory(cate.copy());
		}
		for(int i = 0;i < instances.numInstances();i++) {
			Instance instance = instances.instance(i);
			Integer truevalue = (int)instance.classValue();
			Example example = new Example(instance);
			Label truelabel = new Label(null, truevalue.toString(), example.getId(), "creat");
			example.setTrueLabel(truelabel);
			dataset.addExample(example);
			
		}
		return dataset;
	}
	
}

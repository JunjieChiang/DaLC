package ceka.IWBVT;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import ceka.consensus.MajorityVote;
import ceka.converters.FileLoader;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;

public class Test_T_Ablation {

	private static String dataSetArffDir = "./test/ceka/IWBVT/real-world/";

	private static String[] dataSetArddFix = {"leaves"};

	private static String runDir = "./test/ceka/IWBVT/IWBVT-results/";

	/** the number of instances. */
	private static int num_data = dataSetArddFix.length;

	/** the number of k-fold. */
	private static int num_Folds = 10;

	/** the number of the cross-validation */
	private static int num_times = 10;

	public static void main(String[] args) throws Exception {
		File testDir = new File(runDir);
		if (!testDir.exists())
			testDir.mkdirs();

		// 定义输出格式
		DecimalFormat df = new DecimalFormat("#.00");
		
		// 定义结果文件
		File f = new File(runDir + "leaves_NB_消融.txt");
		FileWriter fw = new FileWriter(f);

		fw.write("dataset\t" + "object1\t" + "object2\t" + "object3\t" + "object4\t");
		fw.write("\r\n");
		fw.flush();

		double all_mvf1 = 0.0;
		double all_mvf2 = 0.0;
		double all_mvf3 = 0.0;
		double all_mvf4 = 0.0;

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
			// 10-runs-10-fold cross-validation
			for (int i = 0; i < num_times; i++) {
				// 复制一个文件防止过拟合
				Dataset temp_data = copyDataset(dataset);

				// 打乱，保证十次flod不同
				Random rand = new Random();
				temp_data.randomize(rand);
				for (int k = 0; k < num_Folds; k++) {
					Dataset train = trainCV(temp_data, num_Folds, k, rand);
					Dataset test = testCV(temp_data, num_Folds, k);
					
					// MV
					Dataset temp_train = copyDataset(train);
					MajorityVote mvf1 = new MajorityVote();
					mvf1.doInference(temp_train);

					// 设置分类器
					Classifier m_classifier1 = new NaiveBayes();
					IWBVT_noIW m_classifier2 = new IWBVT_noIW();
					m_classifier2.setClassifier(m_classifier1);
					IWBVT_noBVT m_classifier3 = new IWBVT_noBVT();
					m_classifier3.setClassifier(m_classifier1);
					IWBVT m_classifier4 = new IWBVT();
					m_classifier4.setClassifier(m_classifier1);
					
					// MV
					m_classifier1.buildClassifier(temp_train);
					Evaluation evaluation = new Evaluation(test);
					evaluation.evaluateModel(m_classifier1, test);
					result_mvf1 += evaluation.pctCorrect();
					
					m_classifier2.buildClassifier2(temp_train);
					evaluation = new Evaluation(test);
					evaluation.evaluateModel(m_classifier2, test);
					result_mvf2 += evaluation.pctCorrect();
					
					m_classifier3.buildClassifier2(temp_train);
					evaluation = new Evaluation(test);
					evaluation.evaluateModel(m_classifier3, test);
					result_mvf3 += evaluation.pctCorrect();
					
					m_classifier4.buildClassifier2(temp_train);
					evaluation = new Evaluation(test);
					evaluation.evaluateModel(m_classifier4, test);
					result_mvf4 += evaluation.pctCorrect();
				}

			}
			String dataName = dataSetArddFix[data].split("/")[0];

			fw.write(dataName + "\t"
					+ df.format(result_mvf1 / num_Folds / num_times) + "\t"
					+ df.format(result_mvf2 / num_Folds / num_times) + "\t"
					+ df.format(result_mvf3 / num_Folds / num_times) + "\t"
					+ df.format(result_mvf4 / num_Folds / num_times) + "\t"
			);

			fw.write("\r\n");
			fw.flush();
			
			all_mvf1 += result_mvf1 / num_Folds / num_times;
			all_mvf2 += result_mvf2 / num_Folds / num_times;
			all_mvf3 += result_mvf3 / num_Folds / num_times;
			all_mvf4 += result_mvf4 / num_Folds / num_times;
		}
		fw.write("average " + "\t" 
				+ df.format(all_mvf1 / num_data) + "\t"
				+ df.format(all_mvf2 / num_data) + "\t"
				+ df.format(all_mvf3 / num_data) + "\t"
				+ df.format(all_mvf4 / num_data) + "\t"
		);

		fw.write("\r\n");
		fw.flush();
		fw.close();
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

	

	public static Dataset copyDataset(Dataset dataset, int start, int end) {
		Dataset copyDataset = new Dataset(dataset, 0);
		for (int k = start; k < end; k++) {
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

	public static Dataset addDataset(Dataset oldDataset, Dataset newDataset) {
		for (int k = 0; k < newDataset.getExampleSize(); k++) {
			Example example = newDataset.getExampleByIndex(k);
			// example.setId(String.valueOf(k + oldDataset.getExampleSize()));
			oldDataset.addExample(example);
		}
		for (int k = 0; k < newDataset.getWorkerSize(); k++) {
			Worker worker = newDataset.getWorkerByIndex(k);
			newDataset.addWorker(worker);
		}
		return oldDataset;
	}

	public double crossValidateModel(Classifier classifier, Dataset dataset,
			int num_Folds, Random random, Object... forPredictionsPrinting)
			throws Exception {
		Random rand = new Random();
		double predict = 0.0;
		for (int i = 0; i < num_Folds; i++) {
			Dataset train = trainCV(dataset, num_Folds, i, rand);
			Dataset test = testCV(dataset, num_Folds, i);
			classifier.buildClassifier(train);
			double m_Incorrect = 0;
			double m_Correct = 0;
			double m_WithClass = 0;
			for (int j = 0; j < test.getExampleSize(); j++) {
				Example example = test.getExampleByIndex(j);
				int actualClass = example.getTrueLabel().getValue();
				m_WithClass += 1;
				double[] distribution = classifier
						.distributionForInstance(example);
				int pred = 0;
				if (distribution[0] < distribution[1]) {
					pred = 1;
				}
				if (actualClass == pred) {
					m_Correct += 1;
				} else {
					m_Incorrect += 1;
				}
			}
			predict += 100 * m_Correct / m_WithClass;
		}
		return predict / num_Folds;
	}

	public static Dataset trainCV(Dataset dataset, int numFolds, int numFold,
			Random random) {

		Dataset train = trainCV(dataset, numFolds, numFold);
		train.randomize(random);
		return train;
	}

	public static Dataset trainCV(Dataset dataset, int numFolds, int numFold) {
		int dataSize = dataset.getExampleSize();
		int numInstForFold, first, offset;
		Dataset train;

		if (numFolds < 2) {
			throw new IllegalArgumentException(
					"Number of folds must be at least 2!");
		}
		if (numFolds > dataset.getExampleSize()) {
			throw new IllegalArgumentException(
					"Can't have more folds than instances!");
		}
		numInstForFold = dataSize / numFolds;
		if (numFold < dataSize % numFolds) {
			numInstForFold++;
			offset = numFold;
		} else {
			offset = dataset.getExampleSize() % numFolds;
		}
		first = numFold * (dataSize / numFolds) + offset;
		train = copyDataset(dataset, 0, first);
		Dataset addData = copyDataset(dataset, first + numInstForFold, dataSize);
		train = addDataset(train, addData);

		return train;
	}

	public static Dataset testCV(Dataset dataset, int numFolds, int numFold) {

		int numInstForFold, first, offset;
		Dataset test;
		int dataSize = dataset.getExampleSize();

		if (numFolds < 2) {
			throw new IllegalArgumentException(
					"Number of folds must be at least 2!");
		}
		if (numFolds > dataSize) {
			throw new IllegalArgumentException(
					"Can't have more folds than instances!");
		}
		numInstForFold = dataSize / numFolds;
		if (numFold < dataSize % numFolds) {
			numInstForFold++;
			offset = numFold;
		} else {
			offset = dataSize % numFolds;
		}
		first = numFold * (dataSize / numFolds) + offset;
		test = copyDataset(dataset, first, first + numInstForFold);
		return test;
	}

	public double getCrossValidateModelEntropy(Classifier m_Classifier,
			Dataset dataset, int num_Folds, Random random) throws Exception {
		Random rand = new Random();
		double Entropy = 0.0;
		for (int i = 0; i < num_Folds; i++) {
			Dataset train = trainCV(dataset, num_Folds, i, rand);
			Dataset test = testCV(dataset, num_Folds, i);
			m_Classifier.buildClassifier(train);
			for (int j = 0; j < test.getExampleSize(); j++) {
				Example example = test.getExampleByIndex(j);
				double[] probs = m_Classifier.distributionForInstance(example);
				Entropy -= (probs[0] * (Math.log(probs[0]))) + (1 - probs[0])
						* (Math.log(1 - probs[0]));
			}
		}
		return Entropy / num_Folds;
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

package ceka.IWBVT;

import java.io.BufferedReader;
import java.io.FileReader;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.utils.DatasetManipulator;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionHandler;
import weka.core.Utils;

public class CekaUtils implements RevisionHandler{
	
	// 生成一个范围内的随机数
	public static double randdouble(double max, double min) {
		return (Math.random() * (max - min) + min);
	}

	//复制数据
	public static Dataset datasetCopy(Dataset dataset) {

		Dataset newdataset = dataset.generateEmpty();

		// 类别
		int numCateSize = dataset.getCategorySize();
		for (int i = 0; i < numCateSize; i++) {
			Category cate = dataset.getCategory(i);
			newdataset.addCategory(cate.copy());
		}

		// 样本
		for (int j = 0; j < dataset.getExampleSize(); j++) {
			Example example = dataset.getExampleByIndex(j);
			newdataset.addExample(example);
		}

		// 工人
		for (int i = 0; i < dataset.getWorkerSize(); i++) {
			newdataset.addWorker(dataset.getWorkerByIndex(i));
		}

		return newdataset;
	}
	
	//复制数据(不复制类别)
		public static Dataset datasetCopyWithoutClass(Dataset dataset) {

			Dataset newdataset = dataset.generateEmpty();

			// 样本
			for (int j = 0; j < dataset.getExampleSize(); j++) {
				Example example = dataset.getExampleByIndex(j);
				newdataset.addExample(example);
			}

			// 工人
			for (int i = 0; i < dataset.getWorkerSize(); i++) {
				newdataset.addWorker(dataset.getWorkerByIndex(i));
			}

			return newdataset;
		}
	
	// 计算集成精度
	public static double integrationAccuracy(Dataset dataset) {

		int numNoisyExample = 0;
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example example = dataset.getExampleByIndex(i);
			if (example.getIntegratedLabel().getValue() != example.getTrueLabel().getValue())
				numNoisyExample++;
		}
		return 100 - (100 * numNoisyExample / (double) dataset.getExampleSize());
	}
	
	// 计算平均集成精度
	public static double integrationAccuracies(Dataset dataset) {
		int numClasses = dataset.getCategorySize();
		double[] temp1 = new double[numClasses];
		double[] temp2 = new double[numClasses];
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example example = dataset.getExampleByIndex(i);
			int classValue = example.getTrueLabel().getValue();
			temp1[classValue] += 1;
			if (example.getIntegratedLabel().getValue() == example.getTrueLabel().getValue())
				temp2[classValue] += 1;
		}
		double temp = 0.0;
		int count = 0;
		for(int k=0; k<numClasses; k++) {
			if(temp1[k] != 0) {
				temp += temp2[k] / temp1[k];
				count += 1;
			}
		}
		return temp / count;
	}
	
	// 计算噪声比
	public static double noiseRatio(Dataset dataset) {

		int numNoisyExample = 0;
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example example = dataset.getExampleByIndex(i);
			if (example.getIntegratedLabel().getValue() != example.getTrueLabel().getValue())
				numNoisyExample++;
		}
		/*return 100 - (100 * numNoisyExample / (double) dataset.getExampleSize());*/
		return (100 * numNoisyExample / (double) dataset.getExampleSize());
	}
	
	//数据集中的噪声个数
	public static double numNoise(Dataset dataset) {

		int numNoisyExample = 0;
		for (int i = 0; i < dataset.getExampleSize(); i++) {
			Example example = dataset.getExampleByIndex(i);
			if (example.getIntegratedLabel().getValue() != example.getTrueLabel().getValue())
				numNoisyExample++;
		}
		return numNoisyExample;
	}
	
	//将instances封装成dataset
	public static Dataset instancesToDataset(Instances instances,Dataset dataset1) {
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
	
	//获取数据集的MV集成标签
	public static int[] getDatasetMVMVIntegratedL(int numExamples, String mvIntegratedLPath) throws Exception {
		// read mvIntegratedL file
		FileReader reader = new FileReader(mvIntegratedLPath);
		BufferedReader readerBuffer = new BufferedReader(reader);
		String line = null;
		int[] mvIntegratedL = new int[numExamples];
		
		int i = 0;
		while((line = readerBuffer.readLine()) != null) {
			String [] subStrs = line.split("[ \t]");
			mvIntegratedL[i] = Integer.parseInt(subStrs[1]);
			i++;
		}
		readerBuffer.close();
		reader.close();
		return mvIntegratedL;
	}
	
	// 十次十折交叉验证，分类精度测试比较的是样本的真实标签,更新版
	public static double classificationAccuracy(Dataset dataset, int times, int nFold, Classifier classifier) throws Exception {
		double acc = 0;
		for (int i = 0; i < times; i++) {
			// 切分为10份
			Dataset[] sumDataset = DatasetManipulator.split(dataset, nFold, true);
			// 选择一份作为测试集，另外9份合并为训练集
			for (int j = 0; j < nFold; j++) {
				int counts = 0;
				Dataset[] trainTestDataset = DatasetManipulator.pickCombine(sumDataset, j);
				classifier.buildClassifier(trainTestDataset[0]);
				for (int k = 0; k < trainTestDataset[1].getExampleSize(); k++) {
					if ((int) classifier.classifyInstance(trainTestDataset[1].instance(k)) == trainTestDataset[1]
							.getExampleByIndex(k).getTrueLabel().getValue()) {
						counts++;
					}
				}
				acc += (double)counts / trainTestDataset[1].numInstances();
			}
		}
		return (acc / times / nFold) * 100;
	}
	
	// 计算F1score(二类直接计算，多类用各类别算数平均)
	public static double calF1(Dataset dataset) throws Exception {
		int numClass = dataset.getCategorySize();
		double[] temp1 = new double[numClass];  // 真实的
		double[] temp2 = new double[numClass];  // 预测的
		double[] temp3 = new double[numClass];  // 估对的
		
		for(int i=0; i<dataset.getExampleSize(); i++) {
			Example example = dataset.getExampleByIndex(i);
			int c1 = example.getTrueLabel().getValue();
			int c2 = example.getIntegratedLabel().getValue();
			temp1[c1] += 1;
			temp2[c2] += 1;
			if(c1 == c2) {
				temp3[c1] += 1;
			}
		}
		
		double F1 = 0.0;
		if(numClass == 2) {
			// 先确定少数类
			int markIndex = 0;
			if(temp1[0] > temp1[1])
				markIndex = 1;
			else
				markIndex = 0;
			F1 = 2 * (temp3[markIndex] / temp2[markIndex] * temp3[markIndex] / temp1[markIndex]) / (temp3[markIndex] / temp2[markIndex] + temp3[markIndex] / temp1[markIndex]);
		}
		else {
			int count = 0;
			for(int k=0; k<numClass; k++) {
				if(temp1[k] != 0) {
					if(temp2[k] != 0) {
						if(temp3[k] != 0)
							F1 += 2 * (temp3[k] / temp2[k] * temp3[k] / temp1[k]) / (temp3[k] / temp2[k] + temp3[k] / temp1[k]);
					}
					count += 1;
				}
			}
			F1 = F1 / count;
		}
		return F1 * 100;
	}
	
	// 输出每一辙的精度，用于t检验
	public static double[] classificationAccuracies(Dataset dataset, int times, int nFold, Classifier classifier) throws Exception {
		double[] acc = new double[times * nFold];
		for (int i = 0; i < times; i++) {
			// 切分为10份
			Dataset[] sumDataset = DatasetManipulator.split(dataset, nFold, true);
			// 选择一份作为测试集，另外9份合并为训练集
			for (int j = 0; j < nFold; j++) {
				int counts = 0;
				Dataset[] trainTestDataset = DatasetManipulator.pickCombine(sumDataset, j);
				classifier.buildClassifier(trainTestDataset[0]);
				for (int k = 0; k < trainTestDataset[1].getExampleSize(); k++) {
					if ((int) classifier.classifyInstance(trainTestDataset[1].instance(k)) == trainTestDataset[1]
							.getExampleByIndex(k).getTrueLabel().getValue()) {
						counts++;
					}
				}
				acc[i * nFold + j] = (double)counts / trainTestDataset[1].numInstances();
			}
		}
		return acc;
	}
	
	// 为BVT构建的分类器的精度
	public static double classificationAccuracyforOneStage(Dataset dataset, int times, int nFold, IWBVT classifier) throws Exception {
		double acc = 0;
		for (int i = 0; i < times; i++) {
			// 切分为10份
			Dataset[] sumDataset = DatasetManipulator.splitforOneStage(dataset, nFold, true);
			// 选择一份作为测试集，另外9份合并为训练集
			for (int j = 0; j < nFold; j++) {
				int counts = 0;
				Dataset[] trainTestDataset = DatasetManipulator.pickCombineforOneStage(sumDataset, j);
				classifier.buildClassifier2(trainTestDataset[0]);
				for (int k = 0; k < trainTestDataset[1].getExampleSize(); k++) {
					if ((int) classifier.classifyInstance(trainTestDataset[1].instance(k)) == trainTestDataset[1]
							.getExampleByIndex(k).getTrueLabel().getValue()) {
						counts++;
					}
				}
				acc += (double)counts / trainTestDataset[1].numInstances();
			}
		}
		return (acc / times / nFold) * 100;
	}
	
	// 对应的精度s
	public static double[] classificationAccuraciesforOneStage(Dataset dataset, int times, int nFold, IWBVT classifier) throws Exception {
		double[] acc = new double[times * nFold];
		for (int i = 0; i < times; i++) {
			// 切分为10份
			Dataset[] sumDataset = DatasetManipulator.splitforOneStage(dataset, nFold, true);
			// 选择一份作为测试集，另外9份合并为训练集
			for (int j = 0; j < nFold; j++) {
				int counts = 0;
				Dataset[] trainTestDataset = DatasetManipulator.pickCombineforOneStage(sumDataset, j);
				classifier.buildClassifier2(trainTestDataset[0]);
				for (int k = 0; k < trainTestDataset[1].getExampleSize(); k++) {
					if ((int) classifier.classifyInstance(trainTestDataset[1].instance(k)) == trainTestDataset[1]
							.getExampleByIndex(k).getTrueLabel().getValue()) {
						counts++;
					}
				}
				acc[i * nFold + j] = (double)counts / trainTestDataset[1].numInstances();
			}
		}
		return acc;
	}
		
	//噪声F1
	public static double[] F1CleanAndRecallNoise(Dataset cleanset, Dataset noiseset) {

		double[] results = new double[3];
		
		int numCleanExampleClean = 0;
		int numNoiseExampleClean = 0;
		for (int i = 0; i < cleanset.getExampleSize(); i++) {
			Example example = cleanset.getExampleByIndex(i);
			if (example.getIntegratedLabel().getValue() == example.getTrueLabel().getValue())
				numCleanExampleClean++;
			else numNoiseExampleClean++;
		}
		
		int numNoiseExampleNoise = 0;
		for (int i = 0; i < noiseset.getExampleSize(); i++) {
			Example example = noiseset.getExampleByIndex(i);
			if (example.getIntegratedLabel().getValue() != example.getTrueLabel().getValue())
				numNoiseExampleNoise++;
		}
		
		results[0] = (double)numNoiseExampleNoise * 100 / (double)noiseset.getExampleSize();
		results[1] = (double)numNoiseExampleNoise * 100 / (double)(numNoiseExampleNoise + numNoiseExampleClean);
		results[2] = (2.0 * (double)numNoiseExampleNoise) / (double)(noiseset.getExampleSize() + cleanset.getExampleSize() + numNoiseExampleNoise - numCleanExampleClean);
		
		return results;
	}
		
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		return null;
	}
	
}

package ceka.consensus.gtic;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import org.apache.log4j.Logger;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.utils.Misc;
import weka.clusterers.ClusterEvaluation;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

public class GTIC {

	private static Logger log = Logger.getLogger(GTIC.class);
	
	/////////////////////////////////////////////////////////////////////////
	// a auxiliary classes
	////////////////////////////////////////////////////////////////////////
	class Group
	{
		public Group(int numClass, int gID)
		{
			id = gID;
			estimatedClassCount = new int [numClass];
			estimatedClassProb = new double [numClass];
			labelCount = new int [numClass];
		}
		
		int id = -1;
		ArrayList<Example>  examples = new ArrayList<Example>();
		int [] estimatedClassCount = null;
		double [] estimatedClassProb = null;
		int [] labelCount = null;
		int estimatedClass = -1;
		
		public void addExample(Example e, Instance inst)
		{
			examples.add(e);
		}
		
		public void statistic()
		{
			class SortElem  implements Comparable<SortElem>
			{
				public int cate = 0;
				public int count = 0;
				public int compareTo(SortElem e) 
				{
					if (count == e.count)
						return 0;
					else if (count > e.count)
						return -1;
					else
						return 1;
				}
			}
			
			for (int i = 0; i < examples.size(); i++)
			{
				ArrayList<SortElem> classCountList = new ArrayList<SortElem>();
				Example example = examples.get(i);
				
			    ArrayList<ArrayList<Label>> labelSorted = statisticMultiClassLabel(labelCount.length, example.getMultipleNoisyLabelSet(0));
				for (int k = 0; k < labelSorted.size(); k++)
				{
					SortElem elem = new SortElem();
					elem.cate = k;
					elem.count = labelSorted.get(k).size();
					classCountList.add(elem);
					// static total count
					labelCount[k] += elem.count;
				}
				
				Collections.sort(classCountList);
				
				ArrayList<SortElem> maxCountList = new ArrayList<SortElem>();
				maxCountList.add(classCountList.get(0));
				for (int j = 1; j < classCountList.size(); j++)
				{
					if (classCountList.get(j).count == classCountList.get(0).count)
						maxCountList.add(classCountList.get(j));
					else
						break;
				}
				int cate = -1;
				if (maxCountList.size() > 1)
				{
					double r = Math.random();
					double grain = 1.0 / (double)maxCountList.size();
					int s =  (int)(r / grain);
					cate = maxCountList.get(s).cate;
				}
				else
				{
					cate = maxCountList.get(0).cate;
				}
				// record category
				estimatedClassCount[cate]++;
			}
			for (int k = 0; k < estimatedClassCount.length; k++)
				estimatedClassProb[k] = (double)estimatedClassCount[k]/(double)examples.size();
		}
		
		public void setEstimatedClass(int c)
		{
			estimatedClass = c;
			for (int i = 0; i < examples.size(); i++)
			{
				Example example = examples.get(i);
				Integer cate = new Integer(estimatedClass);
				Label integratedLabel = new Label(null, cate.toString(), example.getId(), "SimpleKMeans");
				example.setIntegratedLabel(integratedLabel);
			}
		}
		
		private ArrayList<ArrayList<Label>> statisticMultiClassLabel(int numClass, MultiNoisyLabelSet mnls) {
			ArrayList<ArrayList<Label>> labelSorted = new ArrayList<ArrayList<Label>>();
			for (int k = 0; k < numClass; k++)
				labelSorted.add(new ArrayList<Label>());
			for (int i = 0; i < mnls.getLabelSetSize(); i++) {
				Label nL =  mnls.getLabel(i);
				labelSorted.get(nL.getValue()).add(nL);
			}
			return labelSorted;
		}
	}
	
	class WekaInstanceDesc
	{
		public String exampleId;
		
		public ArrayList<Double> labelPercentage = null;
		public ArrayList<Double> betaLabelProb = null;
		public ArrayList<Double> diff = null;
		public Integer trueClass;
		
		public String generateArffInstanceStr(boolean withDiff)
		{
			String instStr = "";
			for (int i = 0; i < labelPercentage.size(); i++)
				instStr += labelPercentage.get(i).toString() + ",";
			if (withDiff)
			{
				for (int i = 0; i < diff.size(); i++)
					instStr += diff.get(i).toString() + ",";
			}
			instStr += (trueClass.toString() + "\n");
			return instStr;
		}
	}
	
	/////////////////////////////////////////////////////////////////////////
	public GTIC( String runDir)
	{
		this.runDir = runDir;
		simpleKMeans = new SimpleKMeansEx();
	}
	
	public long getExcuteTime() {
		return excuteTime;
	}
	
	public int [] getInitialCentroids()
	{
		return initialCentroids;
	}
	
	public void doInference(Dataset dataset)
	{
		this.rootDataset = dataset;
		this.dataName = this.rootDataset.relationName();
		try {
			//Step 1. Generate Features
			generateFeatures(rootDataset);
			//Step 2. Clustering with K-Means
			clustering();
			//Step 3. Class Assignment
			classAssignment();
		} catch (Exception e) {
			e.printStackTrace();
		}
		// this is important
		dataset.assignIntegeratedLabel2WekaInstanceClassValue();
	}
	
	//////////////////////////////////////////////////////
	// generate features
	//////////////////////////////////////////////////////
	protected ArrayList<WekaInstanceDesc> generateFeatures(Dataset dataset) throws IOException
	{
		//ArrayList<Category> categories = rootDataset.getCategories();
		
		arffPath = new String(runDir + dataName + ".arff");
		
		FileWriter arffFile = new FileWriter(new File(arffPath));
		String ralationName = "@relation "+ dataName +"\n";
		arffFile.write(ralationName);
		
		String labelPercentageDesc = generateLabelPercentageDesc(dataset.getCategorySize());arffFile.write(labelPercentageDesc);
		String diffDesc = generateDiffDesc(dataset.getCategorySize()); arffFile.write(diffDesc);
		String classDesc = generateClassDesc(dataset.getCategorySize());
		arffFile.write(classDesc);
		
		arffFile.write("@data\n");
		// generate weka instances
		long startTime = System.nanoTime();
		ArrayList<WekaInstanceDesc> wekaInstances = generateWekaInstances();
		excuteTime += (System.nanoTime() - startTime);
		
		for (int i = 0; i < wekaInstances.size(); i++)
		{
			String dataDesc = wekaInstances.get(i).generateArffInstanceStr(true);
			arffFile.write(dataDesc);
		}
		
		arffFile.close();
		
		return 	wekaInstances;
	}
	
	protected String generateLabelPercentageDesc(int cateNum)
	{
		String featuresDesc = "";
		for (int i = 0; i < cateNum; i++)
		{
			String desc = "@attribute L" + new Integer(i).toString()+ " numeric\n";
			featuresDesc += desc;
		}
		return featuresDesc;
	}
	
	protected String generateDiffDesc(int cateNum)
	{
		return new String("@attribute DIFF numeric\n");
	}
	
	protected String generateClassDesc(int cateNum)
	{
		String featuresDesc = "@attribute class {";
		
		for (int i = 0; i < cateNum; i++)
		{
			if (i != (cateNum - 1))
				featuresDesc += (new Integer(i).toString() + ",");
			else
				featuresDesc += (new Integer(i).toString());
		}
		featuresDesc += "}\n\n";
		
		return featuresDesc;
	}
	
	protected ArrayList<WekaInstanceDesc> generateWekaInstances()
	{
		ArrayList<WekaInstanceDesc> wekaInstances = new ArrayList<WekaInstanceDesc>();
		int cateNum = rootDataset.getCategorySize();
		
		for (int i = 0; i < rootDataset.getExampleSize(); i++)
		{
			Example originalExample = rootDataset.getExampleByIndex(i);
			WekaInstanceDesc instance = new WekaInstanceDesc();
			instance.exampleId = new String(originalExample.getId());
			instance.trueClass = originalExample.getTrueLabel().getValue();
			generateDiffFeature(originalExample, cateNum, instance);
			wekaInstances.add(instance);
		}
		return wekaInstances;
	}
	
	protected ArrayList<Double> generateLabelPercentage(Example example, int cateNum)
	{
		ArrayList<Double> catePercentages = new ArrayList<Double>();
		int [] noisyLabelCount = new int[cateNum];
		double [] labelCategoryPercentage = new double[cateNum];
		int totalNoisyLabelCount = example.getMultipleNoisyLabelSet(0).getLabelSetSize();
		int checkTotal = 0;
		for (int j = 0; j < totalNoisyLabelCount; j++)
		{
			Label noisyLabel = example.getMultipleNoisyLabelSet(0).getLabel(j);
			noisyLabelCount[noisyLabel.getValue()]++;
			checkTotal++;
		}
		assert checkTotal == totalNoisyLabelCount;
		for (int k = 0; k < cateNum; k++)
		{
			labelCategoryPercentage[k] = new Double((double) noisyLabelCount[k] / (double)(totalNoisyLabelCount));
			catePercentages.add(labelCategoryPercentage[k]);
		}
		return catePercentages;
	}
	
	protected void generateDiffFeature(Example example, int cateNum, WekaInstanceDesc wekaInst)
	{
		wekaInst.labelPercentage = generateLabelPercentage(example, cateNum);
		double v = 0.0;
		wekaInst.diff = new ArrayList<Double>();
		for (int k = 0; k < cateNum -1; k++)
			v += wekaInst.labelPercentage.get(k+1) - wekaInst.labelPercentage.get(k);
		wekaInst.diff.add(new Double(v/(double)cateNum));
	}
	
	//////////////////////////////////////////////////////
	// Clustering
	//////////////////////////////////////////////////////
	protected void clustering() throws Exception
	{
		BufferedReader reader;
		reader = new BufferedReader(new FileReader(arffPath));
		clusterData = new Instances(reader);
		reader.close();
		
		long startTime = System.nanoTime();
		clusterData.deleteAttributeAt(clusterData.numAttributes() - 1);
		int numClass = rootDataset.getCategorySize();
		
		// select initial K centroids
		class SortElem  implements Comparable<SortElem>
		{
			public double value = 0;
			public int index = 0;
			public SortElem(int i, double v)
			{
				index = i;
				value = v;
			}
			public int compareTo(SortElem e) 
			{
				if (Misc.isDoubleSame(value, e.value, 0.0000001))
					return 0;
				else if (value > e.value)
					return -1;
				else
					return 1;
			}
		}
		// create K initial centroids
		int [] initialC = new int[numClass];
		ArrayList<ArrayList<SortElem>> cateDistribList = new ArrayList<ArrayList<SortElem>>();
		for (int i = 0; i < numClass; i++)
			cateDistribList.add(new ArrayList<SortElem>());
		for (int i = 0; i < rootDataset.getExampleSize(); i++)
		{
			Example e = rootDataset.getExampleByIndex(i);
			ArrayList<Double> cateLabelProbs = generateLabelPercentage(e, numClass);
			for (int k = 0; k < numClass; k++)
				cateDistribList.get(k).add(new SortElem(i, cateLabelProbs.get(k).doubleValue()));
		}
		for (int k = 0; k < numClass; k++)
		{
			// sort in descending order
			Collections.sort(cateDistribList.get(k));
			// using the first object with maximum value as the centroid
			initialC[k] = cateDistribList.get(k).get(0).index;
		}
		
		// set K classes
		simpleKMeans.setNumClusters(numClass);

		// use Euclidean distance
		EuclideanDistance distanceFun = new EuclideanDistance();		
		String [] distOptions = new String[2];
		StringBuffer optR = new StringBuffer("-R");
		StringBuffer optRnum = new StringBuffer();
		optRnum.append("1-" + clusterData.numAttributes());
		
		distOptions[0] = optR.toString();
		distOptions[1] = optRnum.toString();
		
		distanceFun.setOptions(distOptions);
		distanceFun.setDontNormalize(false);
		
		// set initial centroids
		simpleKMeans.setInitialCentroids(initialC);
		
		// always use default seed by Weka
		simpleKMeans.setSeed(10); 
		simpleKMeans.setDistanceFunction(distanceFun);
		simpleKMeans.setPreserveInstancesOrder(true);
		
		// build Clusterer
		simpleKMeans.buildClusterer(clusterData);
		excuteTime += (System.nanoTime() - startTime);
		
		initialCentroids = simpleKMeans.getInitialCentroids();
		
		// print info.
		String str = simpleKMeans.toString();
		System.out.println(str);	
	}
	
	protected void classAssignment() throws Exception
	{
		long startTime = System.nanoTime();
		log.info("Class Assignment");
		
		int numClass = rootDataset.getCategorySize();
		Group [] groups = new Group[numClass];
		// create K gourps
		for (int k = 0; k < numClass; k++)
			groups[k] = new Group(numClass, k);
		
		// add examples to groups based clustering results
		int[] assignments = simpleKMeans.getAssignments();
		
		for (int i = 0; i < assignments.length; i++)
		{
			Example example = rootDataset.getExampleByIndex(i);
			groups[assignments[i]].addExample(example, clusterData.instance(i));
		}
		
		for (int k = 0; k < numClass; k++)
			groups[k].statistic();
		
		// determine the class label of a group one at a time
		for (int k = 0; k < numClass; k++)
		{
			class Tuple
			{
				Tuple (int g, int c, double p) {
					gId = g;
					cate = c;
				}
				int gId;
				int cate;
			}
			
			double maxProb = -1;
			ArrayList<Tuple> maxProbs = new ArrayList<Tuple>();
			for (int g = 0; g < numClass; g++)
			{
				for (int c = 0; c < numClass; c++)
				{
					if (groups[g].estimatedClassProb[c] != -1) // -1 means invalid
					{
						if (Misc.isDoubleSame(maxProb, groups[g].estimatedClassProb[c], 0.00001))
						{
							maxProbs.add(new Tuple(g, c, groups[g].estimatedClassProb[c]));
						}
						else if (groups[g].estimatedClassProb[c] > maxProb)
						{
							maxProbs.clear();
							maxProbs.add(new Tuple(g, c, groups[g].estimatedClassProb[c]));
							maxProb = groups[g].estimatedClassProb[c];
							
						}
					}
				}
			}
			// if there are multiple elements in maxProbs, use the one with max estimatedClassCount
			int index = 0;
			if (maxProbs.size() > 1)
			{
				int maxCount = 0;
				for (int i = 0; i < maxProbs.size(); i++)
				{
					if (groups[maxProbs.get(i).gId].estimatedClassCount[maxProbs.get(i).cate] > maxCount)
					{
						maxCount = groups[maxProbs.get(i).gId].estimatedClassCount[maxProbs.get(i).cate];
						index = i;
					}
				}
			}
			// set elements in row g or column c to -1, because element in row g and column c is selected 
			for (int g = 0; g < numClass; g++)
			{
				for (int c = 0; c < numClass; c++)
				{
					if ((g == maxProbs.get(index).gId) || (c ==  maxProbs.get(index).cate))
						groups[g].estimatedClassProb[c] = -1;
				}
			}
			// set a group estimated class
			groups[maxProbs.get(index).gId].setEstimatedClass(maxProbs.get(index).cate);
			
			// log info
			log.info("maxProbs (" + maxProbs.size() +") group " + maxProbs.get(index).gId + " <-- " + maxProbs.get(index).cate + "  (class)");
		}
		
		excuteTime += (System.nanoTime() - startTime);
	}
	
	protected String  runDir   = null;
	protected String  dataName = null;
	protected String  arffPath = null;
	protected Dataset rootDataset = null;
	protected Instances clusterData = null;
	protected ClusterEvaluation clusterEvaluation = null;
	protected SimpleKMeansEx simpleKMeans = null;
	protected long    excuteTime = 0;
	protected int [] initialCentroids = null;
}

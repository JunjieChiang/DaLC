package ceka.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;

public class DatasetManipulator {

	/**
	 * split datase to $num sub datasets containing different examples.
	 * @param dataset the dataset is unchanged after being called
	 * @param nFold
	 * @param isShuffled whether randomly shuffle the original dataset 
	 * @return number of $nFold sub datasets
	 */
	public static Dataset [] split(Dataset dataset, int nFold, boolean isShuffled) {
		Dataset [] datasets = new Dataset[nFold];
		int count = dataset.numInstances() / nFold;
		int lastCount = count + dataset.numInstances() % nFold;
		if (isShuffled) {
			Random random = new Random();
			dataset.randomize(random);
		}
		for (int i = 0; i < nFold - 1; i++) {
			datasets[i] = dataset.generateEmpty();
			for (int j = 0; j < count; j++)
				datasets[i].addExample(dataset.getExampleByIndex(i * count + j));
		}
		datasets[nFold - 1]  = dataset.generateEmpty();
		for (int j = 0; j < lastCount; j++) {
			datasets[nFold - 1].addExample(dataset.getExampleByIndex((nFold - 1) * count + j));
		}
		return datasets;
	}
	
	/**
	 * pick up $select dataset from datasets and combine the remaining to a whole one
	 * @param datasets the dataset is unchanged after being called
	 * @param select
	 * @return two datasets, the first is combined one and the second is selected one
	 */
	public static Dataset [] pickCombine(Dataset [] datasets, int select) {
		Dataset [] results = new Dataset[2];
		results[0] = datasets[0].generateEmpty();
		results[1] = datasets[0].generateEmpty();
		for (int i = 0; i < datasets.length; i++)
			if (i != select)
				for (int j = 0; j < datasets[i].numInstances(); j++)
					results[0].addExample(datasets[i].getExampleByIndex(j));
			else
				for (int j = 0; j < datasets[i].numInstances(); j++)
					results[1].addExample(datasets[i].getExampleByIndex(j));
		return results;
	}
	
	/**
	 * add all examples in dataset2 into dataset1
	 * @param ds1 dataset 1
	 * @param ds2 dataset 2
	 */
	public static void addAllExamples(Dataset ds1, Dataset ds2) {
		for (int i = 0; i < ds2.getExampleSize(); i++)
			ds1.addExample(ds2.getExampleByIndex(i));
	}
	
	/**
	 * Split a data set into a training set and a validating set. Instances belonging to each
	 * class in original data set are split at the same ratio, but the instances belonging to 
	 * newly created two data sets are randomly chosen. 
	 * @param dataset original data set
	 * @param cut     the cut point between training set and validation set
	 * @return one training set and one test set 
	 */
	public static Dataset [] splitRandAcrossClass(Dataset dataset, double cut) {
		Dataset [] trainTestDatasets = new Dataset[2];
		
		trainTestDatasets[0] = new Dataset(dataset, 0); // training set
		trainTestDatasets[1] = new Dataset(dataset, 0); // validation set
		
		// see how many classes
		int numCategory = dataset.getCategorySize();
		// copy the categories to two newly created data sets
		for (int i = 0; i < numCategory; i++) {
			Category cate = dataset.getCategory(i);
			trainTestDatasets[0].addCategory(cate.copy());
			trainTestDatasets[1].addCategory(cate.copy());
		}
		// create numCategory example lists
		ArrayList<ArrayList<Example>> exampleLists = new ArrayList<ArrayList<Example>>();
		for (int c = 0; c < numCategory; c++) {
			// process every category
			ArrayList<Example> exampleList = new ArrayList<Example>();
			exampleLists.add(exampleList);
		}
		// sort all examples 
		int numInstance = dataset.numInstances();
		for (int i = 0; i < numInstance; i++) {
			Example example = dataset.getExampleByIndex(i);
			int cate = (int) example.classValue();
			exampleLists.get(cate).add(example);
		}
		
		// split data sets
		for (int c = 0; c < numCategory; c++) {
			ArrayList<List<Example>> cateLists = null;
			cateLists = Misc.splitRandom(exampleLists.get(c), cut);
			for (Example e : cateLists.get(0)) {
				trainTestDatasets[0].addExample((Example)e.copy());
			}
			for (Example e : cateLists.get(1)) {
				trainTestDatasets[1].addExample((Example)e.copy());
			} 
		}
		
		return trainTestDatasets;
	}
}

package ceka.IWBVT;

import java.util.HashMap;

import ceka.core.Dataset;
import ceka.core.Example;
import weka.classifiers.*;
import weka.core.*;

public class IWBVT extends Classifier {

  /** The training instances used for classification. */
  private Instances[] m_Trains;
  
  /** The base classifier to use */
  private Classifier m_Classifier;
  
  /** ÏßÐÔ»Ø¹éÊý×é */
  private LinearRegression[] m_LinearRegressions;
  
  public HashMap<String, Double> MyWeight;

  public void setClassifier(Classifier temp) {
	  m_Classifier = temp;
  }

  public void buildClassifier2(Dataset dataset) throws Exception {
	// »ù±¾±äÁ¿
	int m_numExamples = dataset.getExampleSize();
	int m_numClasses = dataset.getCategorySize();
	
	m_Trains = new Instances[m_numClasses];
	m_LinearRegressions = new LinearRegression[m_numClasses];

	for(int i=0; i<m_numClasses; i++){
		// Lightweight copy: plain Instance objects (feature values only, no label-set deep copy).
		m_Trains[i] = new Instances(dataset, m_numExamples);
		for (int k = 0; k < m_numExamples; k++) {
			Instance src = dataset.instance(k);
			double[] vals = new double[src.numAttributes()];
			for (int a2 = 0; a2 < src.numAttributes(); a2++) {
				vals[a2] = src.value(a2);
			}
			m_Trains[i].add(new Instance(src.weight(), vals));
		}
		int temp_index = m_Trains[i].classIndex();
		Attribute a = new Attribute("newclass");
		m_Trains[i].insertAttributeAt(a, temp_index);
		m_Trains[i].setClass(m_Trains[i].attribute(temp_index));
		m_Trains[i].deleteAttributeAt(temp_index+1);
		m_LinearRegressions[i] = new LinearRegression();
	}
	
	// ÎªÊµÀý·ÖÅä³õÊ¼µÄÈ¨ÖØ
	MyWeight = new HashMap<String, Double>();
		for(int i=0; i<m_numExamples; i++) {
			Example example = dataset.getExampleByIndex(i);
			int classValue = example.getIntegratedLabel().getValue();
			int labelNumber = example.getMultipleNoisyLabelSet(0).getLabelSetSize();
			if(labelNumber <= 0) {
				MyWeight.put(example.getId(), 0.0);
				example.setWeight(0.0);
				continue;
			}
			double[] mark = new double[m_numClasses];
			double tempSum = 0;
		for(int j=0; j<labelNumber; j++) {
			int tempLabel = example.getMultipleNoisyLabelSet(0).getLabel(j).getValue();
			mark[tempLabel] += 1;
			if(tempLabel != classValue)
				tempSum += 1;
		}
		// ¼ÆËãìØ
		double temp = 0.0;
		for(int j=0; j<m_numClasses; j++) {
			if(j != classValue) {
				if(tempSum != 0 && mark[j] != 0) {
					double temp_p = mark[j] / tempSum;
					temp += -1 * temp_p * Math.log(temp_p);
				}
			}
		}
		// ¼ÆËã¼¯³É±ê¼ÇÀà¸ÅÂÊ
		double temp1 = mark[classValue] / labelNumber;
		if(m_numClasses > 2) {
			// ¹éÒ»»¯temp
			temp = temp / Math.log(m_numClasses - 1);
			if(temp != 0)
				MyWeight.put(example.getId(), temp1 * temp);
			else
				MyWeight.put(example.getId(), temp1);
		}
		else
			MyWeight.put(example.getId(), temp1);
		// ÓÃ¸ÅÂÊÖÐµÄ×î´óÖµµ±ÊµÀýµÄÈ¨ÖØ
		example.setWeight(MyWeight.get(example.getId()));
	}

	m_Classifier.buildClassifier(dataset);
	
	//È»ºó¹¹½¨»Ø¹éÈÎÎñ
	int class_index = dataset.classIndex();
	for(int i=0; i<m_numExamples; i++) {
		double[] temp_prob = m_Classifier.distributionForInstance(dataset.getExampleByIndex(i));
		int temp_index = dataset.getExampleByIndex(i).getTrainingLabel();
		for(int j=0; j<m_numClasses; j++){
			if (j == temp_index){
				m_Trains[j].instance(i).setValue(class_index, 1 - temp_prob[j]);
			}
			else{
				m_Trains[j].instance(i).setValue(class_index, 0.0 - temp_prob[j]);
			}
			m_Trains[j].instance(i).setWeight(MyWeight.get(dataset.getExampleByIndex(i).getId()));
		}
	}
	// ÑµÁ·»Ø¹éÄ£ÐÍ
	for(int i=0;i<dataset.numClasses();i++){
		m_LinearRegressions[i].buildClassifier(m_Trains[i]);
	}
  }

  /**
   * Computes class distribution for a test instance.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   */
  public double[] distributionForInstance(Instance instance) throws Exception {	 
	  double[] probs = m_Classifier.distributionForInstance(instance);
	  for(int j=0;j<probs.length;j++) {
		  probs[j] += m_LinearRegressions[j].classifyInstance(instance);
	  }
	  double minmark = probs[Utils.minIndex(probs)];
	  if (minmark<0) {
		  for(int j=0;j<probs.length;j++) {
			  probs[j] -= minmark;
		  }
	  }
	  Utils.normalize(probs);
	  return probs;
  }
  
  public static void main(String[] args) {

    try {
      System.out.println(Evaluation.evaluateModel(new IWBVT(), args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}
}

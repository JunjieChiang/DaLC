package ceka.IWBVT;

import java.util.HashMap;

import ceka.core.Dataset;
import ceka.core.Example;
import weka.classifiers.*;
import weka.core.*;

public class IWBVT_noBVT extends Classifier {

  /** The training instances used for classification. */
  private Instances[] m_Trains;
  
  /** The base classifier to use */
  private Classifier m_Classifier;
  
  /** 线性回归数组 */
  private LinearRegression[] m_LinearRegressions;

  public HashMap<String, Double> MyWeight;

  public void setClassifier(Classifier temp) {
	  m_Classifier = temp;
  }

  public void buildClassifier2(Dataset dataset) throws Exception {
	// 基本变量
	int m_numExamples = dataset.getExampleSize();
	int m_numClasses = dataset.getCategorySize();
	
	m_Trains = new Instances[m_numClasses];
	m_LinearRegressions = new LinearRegression[m_numClasses];
	
	for(int i=0; i<m_numClasses; i++){
		m_Trains[i] = new Instances(dataset);
		int temp_index = m_Trains[i].classIndex();
		Attribute a = new Attribute("newclass");
		m_Trains[i].insertAttributeAt(a, temp_index);
		m_Trains[i].setClass(m_Trains[i].attribute(temp_index));
		m_Trains[i].deleteAttributeAt(temp_index+1);
		m_LinearRegressions[i] = new LinearRegression();
	}
	
	// 为实例分配初始的权重
	MyWeight = new HashMap<String, Double>();
	for(int i=0; i<m_numExamples; i++) {
		Example example = dataset.getExampleByIndex(i);
		int classValue = example.getIntegratedLabel().getValue();
		int labelNumber = example.getMultipleNoisyLabelSet(0).getLabelSetSize();
		double[] mark = new double[m_numClasses];
		double tempSum = 0;
		for(int j=0; j<labelNumber; j++) {
			int tempLabel = example.getMultipleNoisyLabelSet(0).getLabel(j).getValue();
			mark[tempLabel] += 1;
			if(tempLabel != classValue)
				tempSum += 1;
		}
		// 计算熵
		double temp = 0.0;
		for(int j=0; j<m_numClasses; j++) {
			if(j != classValue) {
				if(tempSum != 0 && mark[j] != 0) {
					double temp_p = mark[j] / tempSum;
					temp += -1 * temp_p * Math.log(temp_p);
				}
			}
		}
		// 计算集成标记类概率
		double temp1 = mark[classValue] / labelNumber;
		if(m_numClasses > 2) {
			// 归一化temp
			temp = temp / Math.log(m_numClasses - 1);
			if(temp != 0)
				MyWeight.put(example.getId(), temp1 * temp);
			else
				MyWeight.put(example.getId(), temp1);
		}
		else
			MyWeight.put(example.getId(), temp1);
		// 用概率中的最大值当实例的权重
		example.setWeight(MyWeight.get(example.getId()));
	}

	m_Classifier.buildClassifier(dataset);
  }

  /**
   * Computes class distribution for a test instance.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   */
  public double[] distributionForInstance(Instance instance) throws Exception {	 
	  double[] probs = m_Classifier.distributionForInstance(instance);
	  return probs;
  }
  
  public static void main(String[] args) {

    try {
      System.out.println(Evaluation.evaluateModel(new IWBVT_noBVT(), args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}
}

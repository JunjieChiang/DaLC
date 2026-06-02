
package ceka.IWBVT;

import ceka.core.Dataset;
import weka.classifiers.*;
import weka.core.*;

public class IWBVT_noIW extends Classifier {

  /** The training instances used for classification. */
  private Instances[] m_Trains;
  
  /** The base classifier to use */
  private Classifier m_Classifier;
  
  /** 线性回归数组 */
  private LinearRegression[] m_LinearRegressions;
  
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

	m_Classifier.buildClassifier(dataset);
	
	//然后构建回归任务
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
		}
	}
	// 训练回归模型
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
      System.out.println(Evaluation.evaluateModel(new IWBVT_noIW(), args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}
}

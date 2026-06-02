package ceka.IWBVT;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;

public class LinearRegression extends Classifier{
	//Êý¾Ý³ÉÔ±
	private double[] m_Wb;				//²ÎÊýÊý×é
	
	private Instances m_Instances;		//ÊµÀý¼¯ºÏ
	private int m_NumAtts;				//ÊôÐÔ¸öÊý
	private int m_NumInstances;			//ÊµÀý¸öÊý

	//ÑµÁ·º¯Êý
	public void buildClassifier(Instances train) throws Exception {
		m_Instances = train;
		m_NumAtts = train.numAttributes();
		m_NumInstances = train.numInstances();
		m_Wb = new double[m_NumAtts];
		Matrix Matrix_X = new Matrix(m_NumInstances, m_NumAtts);
		Matrix Matrix_Y = new Matrix(m_NumInstances,1);
		for(int i=0;i<m_NumInstances;i++) {
			Matrix_Y.set(i, 0, train.instance(i).classValue());
			for(int j=0;j<m_NumAtts-1;j++) {
				Matrix_X.set(i, j, train.instance(i).value(j));
			}
			Matrix_X.set(i, m_NumAtts-1, 1);
		}
		//°´ÕÕ×îÐ¡¶þ³ËÇó½âÏßÐÔ»Ø¹é
	    boolean success = true;
	    double ridge = 0.1;
	    Matrix solution = new Matrix(m_NumAtts, 1);
	    do {
	      Matrix ss = Matrix_X.transpose().times(Matrix_X);
	      // ¶Ô½ÇÏß¼ÓÉÏÒ»¸öÖµ±£Ö¤ÂúÖÈ
	      for (int i = 0; i < m_NumAtts; i++)
	        ss.set(i, i, ss.get(i, i) + ridge);
	      Matrix bb = Matrix_X.transpose().times(Matrix_Y);
	      try {
	    	solution = ss.solve(bb);
	        success = true;
	      } 
	      catch (Exception ex) {
	        ridge *= 10;
	        success = false;
	      }
	    } while (!success);
		for(int i=0;i<m_NumAtts;i++) {
			m_Wb[i] = solution.get(i, 0);
		}
	}
	
	//Ô¤²âº¯Êý
	public double classifyInstance(Instance instance) throws Exception {
		double temp = 0;
		for(int i=0;i<m_NumAtts-1;i++) {
			temp += m_Wb[i] * instance.value(i);
		}
		temp += m_Wb[m_NumAtts-1];
		return temp;
	}
	
	//Ö÷º¯Êý
	public static void main(String argv[]) {
		runClassifier(new LinearRegression(), argv);
	}
}
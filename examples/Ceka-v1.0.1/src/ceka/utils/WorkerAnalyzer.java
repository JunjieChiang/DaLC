package ceka.utils;

import java.util.ArrayList;

import ceka.core.Dataset;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;

public class WorkerAnalyzer {

	public class WorkerInfo implements IdDecorated{
		
		public WorkerInfo(String idStr) {
			id = new String(idStr);
		}
		public String getId() {
			return id;
		}
		public void setSpam(boolean flag) {
			isSpam = flag;
		}
		public boolean getSpam() {
			return isSpam;
		}
		private String id;
		private boolean isSpam = false; 
	}
	
	public void initialize(Dataset dataset) {
		data = dataset;
		int numWorker = dataset.getWorkerSize();
		for (int i = 0; i < numWorker; i++) {
			infos.add(new WorkerInfo(dataset.getWorkerByIndex(i).getId()));
		}
	}
	
	public void analyze () {
		analyzeSpam ();
	}
	
	public ArrayList<WorkerInfo> getWorkerInfo() {
		return infos;
	}
	
	private void analyzeSpam () {
		int numWorker = data.getWorkerSize();
		for (int i = 0; i < numWorker; i++) {
			Worker worker = data.getWorkerByIndex(i);
			MultiNoisyLabelSet mnls = worker.getMultipleNoisyLabelSet(0);
			int c = isConsistent(mnls);
			if ((c != 0 ) && (mnls.getLabelSetSize() >= spamThresholdNum)) {
				// spam
				WorkerInfo wi = Misc.getElementById(infos, worker.getId());
				wi.setSpam(true);
				System.out.println("Woker Id = " + wi.getId() + " is spam.");
			}
		}
	}
	
	private int isConsistent(MultiNoisyLabelSet mnls) {
		int ret = -1;
		int numLabel = mnls.getLabelSetSize();
		int preLabel = mnls.getLabel(0).getValue();
		for (int i = 1; i < numLabel; i++) {
			int currLabel = mnls.getLabel(i).getValue();
			if (currLabel != preLabel) {
				ret = 0;
				break;
			}
		}
		if (ret != 0) {
			if (preLabel == 1)
				ret = 1;
		}
		return ret;
	}
	
	private ArrayList<WorkerInfo> infos = new ArrayList<WorkerInfo>();
	private Dataset data = null;
	private int spamThresholdNum = 5;
}

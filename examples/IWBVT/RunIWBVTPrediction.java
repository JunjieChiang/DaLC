package ceka.IWBVT;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import ceka.consensus.MajorityVote;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.core.Example;
import weka.classifiers.bayes.NaiveBayes;

public class RunIWBVTPrediction {
    private static final class PredictionRow {
        final String objectId;
        final int truth;
        final int pred;

        PredictionRow(String objectId, int truth, int pred) {
            this.objectId = objectId;
            this.truth = truth;
            this.pred = pred;
        }
    }

    private static int compareId(String left, String right) {
        try {
            return Integer.compare(Integer.parseInt(left), Integer.parseInt(right));
        } catch (NumberFormatException ex) {
            return left.compareTo(right);
        }
    }

    private static File findDatasetFile(File datasetDir, String... suffixes) {
        File[] files = datasetDir.listFiles();
        if (files == null) {
            throw new IllegalArgumentException("Dataset directory is unreadable: " + datasetDir.getPath());
        }
        for (String suffix : suffixes) {
            for (File file : files) {
                if (file.isFile() && file.getName().toLowerCase().endsWith(suffix)) {
                    return file;
                }
            }
        }
        throw new IllegalArgumentException("Missing dataset file in " + datasetDir.getPath());
    }

    private static Dataset loadDataset(File datasetDir) throws Exception {
        File arffFile = findDatasetFile(datasetDir, ".arffx", ".arff");
        File responseFile = findDatasetFile(datasetDir, ".response.txt");
        File goldFile = findDatasetFile(datasetDir, ".gold.txt");
        Dataset dataset;
        if (arffFile.getName().toLowerCase().endsWith(".arffx")) {
            dataset = FileLoader.loadFileX(responseFile.getPath(), goldFile.getPath(), arffFile.getPath());
        } else {
            dataset = FileLoader.loadFile(responseFile.getPath(), goldFile.getPath(), arffFile.getPath());
        }
        return replaceMissingValues(dataset);
    }

    // Mirrors WEKA's ReplaceMissingValues filter: numeric -> mean, nominal -> mode.
    private static Dataset replaceMissingValues(Dataset dataset) {
        int numAtts = dataset.numAttributes();
        int classIdx = dataset.classIndex();
        double[] replacements = new double[numAtts];
        for (int i = 0; i < numAtts; i++) {
            if (i != classIdx) {
                replacements[i] = dataset.meanOrMode(i);
            }
        }
        for (int i = 0; i < dataset.numInstances(); i++) {
            for (int j = 0; j < numAtts; j++) {
                if (j != classIdx && dataset.instance(i).isMissing(j)) {
                    dataset.instance(i).setValue(j, replacements[j]);
                }
            }
        }
        return dataset;
    }

    private static void writePredictionCsv(Dataset dataset, int[] predictions, File outputFile) throws Exception {
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        ArrayList<PredictionRow> rows = new ArrayList<PredictionRow>();
        for (int i = 0; i < dataset.getExampleSize(); i++) {
            Example example = dataset.getExampleByIndex(i);
            rows.add(new PredictionRow(example.getId(), example.getTrueLabel().getValue(), predictions[i]));
        }
        Collections.sort(
            rows,
            new Comparator<PredictionRow>() {
                public int compare(PredictionRow left, PredictionRow right) {
                    return compareId(left.objectId, right.objectId);
                }
            }
        );
        PrintWriter writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"));
        try {
            writer.println("object,truth,pred");
            for (PredictionRow row : rows) {
                writer.println(row.objectId + "," + row.truth + "," + row.pred);
            }
        } finally {
            writer.close();
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            throw new IllegalArgumentException("Usage: RunIWBVTPrediction <dataset_dir> <output_prediction_csv>");
        }
        File datasetDir = new File(args[0]);
        File outputFile = new File(args[1]);

        Dataset dataset = loadDataset(datasetDir);
        MajorityVote majorityVote = new MajorityVote();
        majorityVote.doInference(dataset);

        IWBVT model = new IWBVT();
        model.setClassifier(new NaiveBayes());
        model.buildClassifier2(dataset);

        int[] predictions = new int[dataset.getExampleSize()];
        for (int i = 0; i < dataset.getExampleSize(); i++) {
            predictions[i] = (int) model.classifyInstance(dataset.getExampleByIndex(i));
        }
        writePredictionCsv(dataset, predictions, outputFile);
    }
}


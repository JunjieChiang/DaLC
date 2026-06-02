package ceka.integration;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

import ceka.consensus.MajorityVote;
import ceka.consensus.ds.DawidSkene;
import ceka.consensus.gtic.GTIC;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.core.Example;

public class RunAggregationPrediction {
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
        if (arffFile.getName().toLowerCase().endsWith(".arffx")) {
            return FileLoader.loadFileX(responseFile.getPath(), goldFile.getPath(), arffFile.getPath());
        }
        return FileLoader.loadFile(responseFile.getPath(), goldFile.getPath(), arffFile.getPath());
    }

    private static void runMethod(String method, Dataset dataset, File workDir, int dsIterations) throws Exception {
        if (!workDir.exists()) {
            workDir.mkdirs();
        }
        if ("MV".equals(method)) {
            MajorityVote majorityVote = new MajorityVote();
            majorityVote.doInference(dataset);
            return;
        }
        if ("DS".equals(method)) {
            DawidSkene dawidSkene = new DawidSkene(dsIterations);
            dawidSkene.doInference(dataset);
            return;
        }
        if ("GTIC".equals(method)) {
            GTIC gtic = new GTIC(workDir.getPath());
            gtic.doInference(dataset);
            return;
        }
        throw new IllegalArgumentException("Unsupported aggregation method: " + method);
    }

    private static void writePredictionCsv(Dataset dataset, File outputFile) throws Exception {
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        ArrayList<PredictionRow> rows = new ArrayList<PredictionRow>();
        for (int i = 0; i < dataset.getExampleSize(); i++) {
            Example example = dataset.getExampleByIndex(i);
            rows.add(
                new PredictionRow(
                    example.getId(),
                    example.getTrueLabel().getValue(),
                    example.getIntegratedLabel().getValue()
                )
            );
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
        if (args.length < 4) {
            throw new IllegalArgumentException(
                "Usage: RunAggregationPrediction <method> <dataset_dir> <output_prediction_csv> <work_dir> [ds_iterations]"
            );
        }
        String method = args[0];
        File datasetDir = new File(args[1]);
        File outputFile = new File(args[2]);
        File workDir = new File(args[3]);
        int dsIterations = args.length >= 5 ? Integer.parseInt(args[4]) : 20;

        Dataset dataset = loadDataset(datasetDir);
        runMethod(method, dataset, workDir, dsIterations);
        writePredictionCsv(dataset, outputFile);
    }
}

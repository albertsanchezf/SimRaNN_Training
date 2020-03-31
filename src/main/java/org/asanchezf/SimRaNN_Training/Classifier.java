/* *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.asanchezf.SimRaNN_Training;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class Classifier {

    static Logger log = Logger.getLogger(Classifier.class);

    public static void main(String[] args) throws  Exception {

        String configFilename = System.getProperty("user.dir")
                + File.separator + "log4j.properties";
        PropertyConfigurator.configure(configFilename);

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        char delimiter = ',';

        String datasetPath = "/Users/AlbertSanchez/Desktop/TFM (noDropBox)/Dataset/binaryDS/dataset.csv"; //All
        String modelConfigPath = "/Users/AlbertSanchez/Dropbox/TFM/OptimizationNetworks/binaryDS/0/modelConfig.json"; //All
        String saveFilename = "resources/trainedNN/DSNet.zip";
        int numClasses = 2;  //2 classes (types of incidents). 0 - No incident | 1 - Incident
        int batchSize = 512; //SimRa dataset: 3896

        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new File(datasetPath)));

        // Build a Input Schema
        Schema inputDataSchema = new Schema.Builder()
                .addColumnsFloat("speed","mean_acc_x","mean_acc_y","mean_acc_z","std_acc_x","std_acc_y","std_acc_z")
                .addColumnDouble("sma")
                .addColumnFloat("mean_svm")
                .addColumnsDouble("entropyX","entropyY","entropyZ")
                .addColumnsInteger("bike_type","phone_location","incident_type")
                .build();

        // Made the necessary transformations
        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .integerToOneHot("bike_type",0,8)
                .integerToOneHot("phone_location",0,6)
                //.removeColumns("speed")
                //.removeColumns("mean_acc_x")
                //.removeColumns("mean_acc_y")
                //.removeColumns("mean_acc_z")
                //.removeColumns("std_acc_x")
                //.removeColumns("std_acc_y")
                //.removeColumns("std_acc_z")
                //.removeColumns("sma")
                //.removeColumns("mean_svm")
                //.removeColumns("entropyX")
                //.removeColumns("entropyY")
                //.removeColumns("entropyZ")
                //.removeColumns("bike_type")
                //.removeColumns("phone_location")
                //.removeColumns("incident_type")
                .build();

        // Get output schema
        Schema outputSchema = tp.getFinalSchema();

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = outputSchema.getColumnNames().size() - 1;     //15 values in each row of the dataset.csv; CSV: 14 input features followed by an integer label (class) index. Labels are the 15th value (index 14) in each row

        TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,tp);
        DataSetIterator iterator = new RecordReaderDataSetIterator(transformProcessRecordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.7);  //Use 70% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainingData);     //Apply normalization to the training data
        normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

        String json = textfile2String(modelConfigPath);

        log.info("Build model....");
        MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(json);
        // Run the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        for(int i=0; i<2000; i++) {
            model.fit(trainingData);
        }

        // Evaluate the model on the test set
        Evaluation eval = new Evaluation(numClasses);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats(true));
        System.out.println(eval.stats());

        // Save the trained model
        File locationToSave; //Where to save the network. Note: the file is in .zip format - can be opened externally
        locationToSave = new File(saveFilename); //All
        renameDSNetLastFile(locationToSave.getAbsolutePath());

        ModelSerializer.writeModel(model,locationToSave, false);
        System.out.println("Model trained saved in: " + locationToSave.toString());

        // Save the statistics for the DS
        ModelSerializer.addNormalizerToModel(locationToSave,normalizer);
        System.out.println("Normalizer statistics saved in the model");

    }

    public static void renameDSNetLastFile(String fullPath)
    {
        File f;
        String[] s, s1, splitpath;
        String filename = "", path = "";
        boolean dsFile = false;
        int maxFile = 0;

        splitpath = fullPath.split("/");
        filename = splitpath[splitpath.length-1].replace(".zip","");
        path = fullPath.substring(0,fullPath.length()-filename.length()-4);
        f = new File(fullPath);

        if (f.exists())
        {
            f = new File(path);
            // To ensure any file is overwritten
            String[] files = f.list();
            for (String file : files)
            {
                if(new File(path+file).isFile())
                {
                    s = file.split(filename);
                    if(s.length==2)
                    {
                        if(!s[1].equals(".zip"))
                        {
                            s1 = s[1].split(".zip");
                            if (s1[0].substring(1).matches("^[0-9]*$"))
                                if(Integer.valueOf(s1[0].substring(1)) > maxFile) maxFile = Integer.valueOf(s1[0].substring(1));
                        }
                    }
                }
            }
            maxFile++;
            new File(path + filename + ".zip").renameTo(new File(path + filename + "-" + String.valueOf(maxFile) + ".zip"));
        }

    }

    private static String textfile2String(String filePath)
    {
        StringBuilder contentBuilder = new StringBuilder();
        try (Stream<String> stream = Files.lines( Paths.get(filePath), StandardCharsets.US_ASCII))
        {
            stream.forEach(s -> contentBuilder.append(s).append("\n"));
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
        return contentBuilder.toString();
    }

}


package org.example;

import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import org.example.LinearModelFeatureImportances;

public class Main {
    public static void main(String[] args) throws Exception {
        String LinearModelPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\LinearModel";
        String LinearDataPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\LinearData";
        String ForestModelPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\ForestModel";

        SparkSession spark = SparkSession.builder().appName("Feature").master("local").getOrCreate();
//        读本地模型
        LinearRegressionModel LinearModel = LinearRegressionModel.load(LinearModelPath);
        RandomForestRegressionModel ForestModel = RandomForestRegressionModel.load(ForestModelPath);
//        读本地数据集
        Dataset<Row> dataset = spark.read()
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(LinearDataPath);

        LinearModelFeatureImportances LinearExecutor = new LinearModelFeatureImportances();
        RandomForestFeatureImportances ForestExecutor = new RandomForestFeatureImportances();
//        线性模型的map
        HashMap<String, Object> map1 = new HashMap<>();
        map1.put("dataset", dataset);
        map1.put("model", LinearModel);
//        String[] featureNames1 = new String[]{"feature1", "feature2"};
        String[] featureNames1 = new String[]{"feature2"};
        map1.put("featureNames", featureNames1);

//        随机森林模型的map
        HashMap<String, Object> map2 = new HashMap<>();
        map2.put("dataset", dataset);
        map2.put("model", ForestModel);
        String[] featureNames2 = new String[]{"feature1", "feature2"};
//        String[] featureNames2 = new String[]{"feature2"};
        map2.put("featureNames", featureNames2);


        LinearExecutor.execute(map1);
        ForestExecutor.execute(map2);

        spark.stop();
    }
}
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
import scala.Tuple2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class Test {
    static List<Row> data = Arrays.asList(
            RowFactory.create(1.0, 2.0, 3.0),
            RowFactory.create(2.0, 4.0, 6.0),
            RowFactory.create(3.0, 6.0, 9.0)
    );

    static List<String> featureNames = Arrays.asList("feature1", "feature2");


    public static String LinearModelPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\LinearModel";
    public static String LinearDataPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\LinearData";
    public static String ForestModelPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\ForestModel";
    public static String ForestDataPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\ForestData";

    public static void saveLinearModel(@NotNull SparkSession spark) {

        // 定义schema
        StructType schema = new StructType()
                .add("feature1", DataTypes.DoubleType)
                .add("feature2", DataTypes.DoubleType)
                .add("label", DataTypes.DoubleType);

        Dataset<Row> dataset = spark.createDataFrame(data, schema);
        // 保存数据集为CSV
        dataset.write().option("header", "true").mode("overwrite").csv(LinearDataPath);

        // 训练一个线性回归模型
        // 注意这里我们现在使用两个特征
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature1", "feature2"})
                .setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(dataset);

        LinearRegression lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features");
        LinearRegressionModel model = lr.fit(assembledData);

        try {
            model.write().overwrite().save(LinearModelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void saveRandomForestModel(@NotNull SparkSession spark) {
        // 这部分与之前相同，用于数据准备
        StructType schema = new StructType()
                .add("feature1", DataTypes.DoubleType)
                .add("feature2", DataTypes.DoubleType)
                .add("label", DataTypes.DoubleType);
        Dataset<Row> dataset = spark.createDataFrame(data, schema);

        // 保存数据集为CSV
        dataset.write().option("header", "true").mode("overwrite").csv(ForestDataPath);

        // 将特性转换为向量格式，这与之前的代码相同
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature1", "feature2"})
                .setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(dataset);

        // 训练一个随机森林模型
        RandomForestRegressor rfr = new RandomForestRegressor()
                .setLabelCol("label")
                .setFeaturesCol("features");
        RandomForestRegressionModel model = rfr.fit(assembledData);

        // 保存随机森林模型
        try {
            model.write().overwrite().save(ForestModelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void printLinearModelFeatureImportances(@NotNull HashMap<String, Object> hashMap) {
        // 1. 加载模型
        LinearRegressionModel model = LinearRegressionModel.load(LinearModelPath);

        // 2. 获取模型的系数
        double[] coefficients = model.coefficients().toArray();

        // 3. 由于我们知道特征名称，我们可以直接使用它们。但在实际应用中，您可能需要从某处获取这些名称。
//        String[] featureNames = {"feature1", "feature2"}; // 根据您的模型和数据集调整
        Dataset<Row> dataset = (Dataset<Row>) hashMap.get("dataset");
        String[] featureNames = dataset.columns();

        // 4. 创建一个列表来存储特征名称和它们的系数
        List<Tuple2<String, Double>> featureImportances = new ArrayList<>();
        for (int i = 0; i < coefficients.length; i++) {
            featureImportances.add(new Tuple2<>(featureNames[i], coefficients[i]));
        }

        // 5. 根据系数的绝对值对特征进行排序
        featureImportances.sort((t1, t2) -> Double.compare(Math.abs(t2._2), Math.abs(t1._2)));

        // 6. 打印排序后的特征重要性
        for (Tuple2<String, Double> tuple : featureImportances) {
            System.out.println("Feature: " + tuple._1 + ", Importance: " + tuple._2);
        }
    }

    public static void printRandomForestFeatureImportances(@NotNull HashMap<String, Object> hashMap) {
        // 1. 加载模型
        RandomForestRegressionModel model = RandomForestRegressionModel.load(ForestModelPath);

        // 2. 获取模型的特征重要性
        double[] importances = model.featureImportances().toArray();

//        // 3. 由于我们知道特征名称，我们可以直接使用它们。但在实际应用中，您可能需要从某处获取这些名称。
//        String[] featureNames = {"feature1", "feature2"}; // 根据您的模型和数据集调整
        Dataset<Row> dataset = (Dataset<Row>) hashMap.get("dataset");
        String[] featureNames = dataset.columns();

        // 4. 创建一个列表来存储特征名称和它们的重要性
        List<Tuple2<String, Double>> featureImportancesList = new ArrayList<>();
        for (int i = 0; i < importances.length; i++) {
            featureImportancesList.add(new Tuple2<>(featureNames[i], importances[i]));
        }

        // 5. 根据重要性对特征进行排序
        featureImportancesList.sort((t1, t2) -> Double.compare(t2._2, t1._2));

        // 6. 打印排序后的特征重要性
        for (Tuple2<String, Double> tuple : featureImportancesList) {
            System.out.println("Feature: " + tuple._1 + ", Importance: " + tuple._2);
        }
    }


    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder().appName("Feature").master("local").getOrCreate();

        Dataset<Row> dataset = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(LinearDataPath);

        HashMap<String, Object> hashMap = new HashMap<>();
        hashMap.put("dataset", dataset);

        String[] columnNames = dataset.columns();

//        saveLinearModel(spark);
//        saveRandomForestModel(spark);
        printLinearModelFeatureImportances(hashMap);
        printRandomForestFeatureImportances(hashMap);


        spark.stop();
    }
}

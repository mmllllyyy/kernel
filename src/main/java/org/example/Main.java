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

public class Main {
    static List<Row> data = Arrays.asList(
            RowFactory.create(1.0, 2.0),
            RowFactory.create(2.0, 4.0),
            RowFactory.create(3.0, 6.0)
    );

    public static String LinearModelPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\LinearModel";
    public static String LinearDataPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\LinearData";
    public static String ForestModelPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\ForestModel";
    public static String ForestDataPath = "C:\\Users\\dwc20\\source\\repos\\kernel\\ForestData";

    public static void saveLinearModel(@NotNull SparkSession spark) {
//        SparkSession spark = SparkSession.builder().appName("Save Model").master("local").getOrCreate();

        StructType schema = new StructType()
                .add("feature", DataTypes.DoubleType)
                .add("label", DataTypes.DoubleType);
        Dataset<Row> dataset = spark.createDataFrame(data, schema);
        // 保存数据集为CSV
        dataset.write().option("header", "true").mode("overwrite").csv(LinearDataPath);

        //        训练一个线性回归模型
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature"})
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

    public static @NotNull HashMap<String, Object> LinearEstimate(@NotNull HashMap<String, Object> params) {
        //        初始化Spark:
//        SparkSession spark = SparkSession.builder().appName("Feature Importance").master("local").getOrCreate();

        //        读取数据集
        Dataset<Row> dataset = (Dataset<Row>) params.get("dataset");
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature"})
                .setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(dataset);

        //        读取模型
        LinearRegressionModel model = LinearRegressionModel.load(LinearModelPath);
        //          预测数据集
        Dataset<Row> predictions = model.transform(assembledData);

        //          评估模型
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(predictions);
        evaluator.setMetricName("r2");
        double r2 = evaluator.evaluate(predictions);

        HashMap<String, Object> output = new HashMap<>();
        output.put("RMSE", rmse);
        output.put("R2", r2);

        return output;
    }


    public static void saveRandomForestModel(@NotNull SparkSession spark) {
        // 这部分与之前相同，用于数据准备
        StructType schema = new StructType()
                .add("feature", DataTypes.DoubleType)
                .add("label", DataTypes.DoubleType);
        Dataset<Row> dataset = spark.createDataFrame(data, schema);

        // 保存数据集为CSV
        dataset.write().option("header", "true").mode("overwrite").csv(ForestDataPath);

        // 将特性转换为向量格式，这与之前的代码相同
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature"})
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

    public static @NotNull HashMap<String, Object> RandomForestEstimate(@NotNull HashMap<String, Object> params) {
        // 使用提供的数据集
        Dataset<Row> dataset = (Dataset<Row>) params.get("dataset");
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature"})
                .setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(dataset);

        // 加载随机森林模型
        RandomForestRegressionModel model = RandomForestRegressionModel.load(ForestModelPath);  // 确保ForestModelPath是保存随机森林模型的正确路径

        // 预测数据集
        Dataset<Row> predictions = model.transform(assembledData);

        // 评估模型
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(predictions);
        evaluator.setMetricName("r2");
        double r2 = evaluator.evaluate(predictions);

        HashMap<String, Object> output = new HashMap<>();
        output.put("RMSE", rmse);
        output.put("R2", r2);

        return output;
    }


    public static Vector getLogisticModelFeatureImportances() {
        LogisticRegressionModel model = LogisticRegressionModel.load(LinearModelPath);
        return model.coefficients();
    }

    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder().appName("Feature").master("local").getOrCreate();


//        saveLinearModel(spark);
//        saveRandomForestModel(spark);

        Dataset<Row> dataset = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(LinearDataPath);

        //  创建参数HashMap并传入数据集
        HashMap<String, Object> params = new HashMap<>();
        params.put("dataset", dataset);

        // 调用评估函数
        HashMap<String, Object> evaluationResults = LinearEstimate(params);
        System.out.println(evaluationResults);

        evaluationResults = RandomForestEstimate(params);
        System.out.println(evaluationResults);

        spark.stop();
    }
}
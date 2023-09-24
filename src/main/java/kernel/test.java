package kernel;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.streaming.state.package$;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class test {
    static List<Row> data = Arrays.asList(
            RowFactory.create(1.0, 2.0),
            RowFactory.create(2.0, 4.0),
            RowFactory.create(3.0, 6.0)
    );
    public static String LinearModelPath = "C:\\Users\\dwc20\\source\\repos\\spark_test\\LinearModel";
    public static String LinearDataPath = "C:\\Users\\dwc20\\source\\repos\\spark_test\\LinearData";
    public static String ForestModelPath = "C:\\Users\\dwc20\\source\\repos\\spark_test\\ForestModel";
    public static String ForestDataPath = "C:\\Users\\dwc20\\source\\repos\\spark_test\\ForestData";

    public static void saveLinearModel(@NotNull SparkSession spark){
        //        初始化Spark:
//        SparkSession spark = SparkSession.builder().appName("Feature Importance").master("local").getOrCreate();
//        Dataset<Row> data = spark.read().format("csv").option("header", "true").load("path_to_your_data.csv");

//        创建一个模拟的数据集:
        StructType schema = new StructType()
                .add("feature", DataTypes.DoubleType)
                .add("label", DataTypes.DoubleType);
        Dataset<Row> dataset = spark.createDataFrame(data, schema);
        dataset.write().option("header", "true").csv(LinearDataPath);

//        训练一个线性回归模型:
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature"})
                .setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(dataset);
//        assembledData.write().option("header", "true").csv("C:\\Users\\dwc20\\source\\repos\\spark_test\\data");

        LinearRegression lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features");
        LinearRegressionModel model = lr.fit(assembledData);

//        保存模型:
        try {
            model.write().overwrite().save(LinearModelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void LinearEstimate() {
//        初始化Spark:
        SparkSession spark = SparkSession.builder().appName("Feature Importance").master("local").getOrCreate();

//        建立模型并保存模型和数据集
//        saveLinearModel(spark);

//        读取数据集
//        Dataset<Row> dataset = spark.read().option("header", "true").csv("C:\\Users\\dwc20\\source\\repos\\spark_test\\data");
        Dataset<Row> dataset = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(LinearDataPath);
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
                .setMetricName("rmse");  // 这里是评估模型的RMSE. 您可以更改为"mae"或"r2"等其他值。

        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

//          如果您想评估其他指标，只需更改setMetricName的参数值即可。
//          例如，要评估R2，您可以这样做：
        evaluator.setMetricName("r2");
        double r2 = evaluator.evaluate(predictions);
        System.out.println("R2 on test data = " + r2);

        spark.stop();
    }

    public static void saveRandomForestModel(SparkSession spark){
        StructType schema = new StructType()
                .add("feature", DataTypes.DoubleType)
                .add("label", DataTypes.DoubleType);
        Dataset<Row> dataset = spark.createDataFrame(data, schema);
        dataset.write().option("header", "true").csv(ForestDataPath);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature"})
                .setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(dataset);

        RandomForestRegressor rf = new RandomForestRegressor()
                .setLabelCol("label")
                .setFeaturesCol("features");
        RandomForestRegressionModel model = rf.fit(assembledData);

        try {
            model.write().overwrite().save(ForestModelPath);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void RandomForestEstimate() {
        SparkSession spark = SparkSession.builder().appName("RandomForest Feature Importance").master("local").getOrCreate();

        saveRandomForestModel(spark);

        Dataset<Row> dataset = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(ForestDataPath);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"feature"})
                .setOutputCol("features");
        Dataset<Row> assembledData = assembler.transform(dataset);

        RandomForestRegressionModel model = RandomForestRegressionModel.load(ForestModelPath);

        Dataset<Row> predictions = model.transform(assembledData);

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");

        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        evaluator.setMetricName("r2");
        double r2 = evaluator.evaluate(predictions);
        System.out.println("R2 on test data = " + r2);

        // 获取特征重要性
        double[] featureImportances = model.featureImportances().toArray();
        for (int i = 0; i < featureImportances.length; i++) {
            System.out.println("Feature " + i + ": " + featureImportances[i]);
        }

        spark.stop();
    }

    public static void main(String[] args) {
        RandomForestEstimate();
    }

}

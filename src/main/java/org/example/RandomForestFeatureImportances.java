package org.example;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import scala.Tuple2;

import java.util.*;

public class RandomForestFeatureImportances extends Operator {
    @Override
    public HashMap<String, Object> execute(HashMap<String, Object> map) throws Exception {
//        PortBean portBean = (PortBean) map.get("model");//模型 out.put("model", new PortBean("model", modelType, pipelineModel, modelInsight, this.dataset.schema()));
//        LinearRegressionModel model = (LinearRegressionModel) portBean.getData();
//         读模型
        RandomForestRegressionModel model = (RandomForestRegressionModel) map.get("model");

//         读数据集
        Dataset dataset = (Dataset) map.get("dataset");

//         读metadata
//        JSONArray metadata = JSONArray.parseArray((String) map.get("metadata")); //[{"colName":"","colType":"","role":"因变量/自变量"}]

//        获取要计算的特征名称
        String[] featureNames = (String[]) map.get("featureNames");

//        执行算子获得特征重要性
        JSONArray jsonArray = printRandomForestFeatureImportances(dataset, model, featureNames);

        HashMap<String, Object> out = new HashMap<>();
        out.put("dataset", dataset);//dataset
        out.put("featureNames", jsonArray);//特征
        return out;
    }
    public JSONArray printRandomForestFeatureImportances(Dataset<Row> dataset, RandomForestRegressionModel model, String[] featureNames) {
        // 获取模型的系数
        // 系数顺序对应训练阶段使用的特征名称顺序，即VectorAssembler的setInputCols顺序
        double[] importances = model.featureImportances().toArray();
        // 获取全部的特征名称与系数对应
        String[] allFeatureNames = dataset.columns();

//         创建一个列表来存储特征名称和它们的系数
        List<Tuple2<String, Double>> featureImportances = new ArrayList<>();
//        使用一个HashSet快速查询featureNames中的项
        Set<String> featureNameSet = new HashSet<>(Arrays.asList(featureNames));
        for (int i = 0; i < importances.length; i++) {
            if (featureNameSet.contains(allFeatureNames[i])) {
                featureImportances.add(new Tuple2<>(allFeatureNames[i], importances[i]));
            }
        }

        // 根据系数的绝对值对特征进行排序
        featureImportances.sort((t1, t2) -> Double.compare(t2._2, t1._2));

        // 存json
        JSONArray jsonArray = new JSONArray();
        for (Tuple2<String, Double> tuple : featureImportances) {
            JSONObject jsonObject = new JSONObject();
            jsonObject.put("Importance", tuple._2);
            jsonObject.put("Feature", tuple._1);
            jsonArray.add(jsonObject);
        }
        System.out.println(jsonArray.toJSONString());

        return jsonArray;
    }
}

package org.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.HashMap;

public abstract class Operator {

    public static Dataset<Row> read_csv(SparkSession sparkSession, String path) {
        return sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(path);
    }

    public abstract HashMap<String, Object> execute(HashMap<String, Object> map) throws Exception;
}

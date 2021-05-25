package org.apache.spark.ml.sampling

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{IndexToString, MinMaxScaler, StringIndexer}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.ml.sampling.Utilities.{convertFeaturesToVector, getCountsByClass}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types._

//FIXME - turn classes back to Ints instead of Doubles
object SamplingExample {

  type NearestClassResult = (Int, Array[Int]) //class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) //index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)

  val asDense: UserDefinedFunction = udf((v: Any) => {
    if (v.isInstanceOf[SparseVector]) {
      v.asInstanceOf[SparseVector].toDense
    } else {
      v.asInstanceOf[DenseVector]
    }
  })

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val labelColumnName = "label"
    val enableDataScaling = true
    val savePath = "/tmp/oversampleX/"
    val input_file = "/home/ford/data/covtype/covtype1k.csv"

    val df = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv(input_file).withColumnRenamed(labelColumnName, "label")

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    val converted: DataFrame = convertFeaturesToVector(results)

    val trainData: DataFrame = if (enableDataScaling) {
      val scaler = new MinMaxScaler().setMin(0.0).setMax(1.0)
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)

      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.drop("features").withColumn("features", asDense($"scaledFeatures")).drop("scaledFeatures")
    } else {
      converted
    }.cache()

    getCountsByClass("label", trainData).show()

    val samplingMap: Map[String, Double] = Map( "5" -> 10.0, "1" -> 10.0 )

    val r = new SMOTE().setSamplingRatios(samplingMap)
    val model = r.fit(trainData)
    val sampledData = model.transform(trainData)

    getCountsByClass("label", sampledData).show()

    def vecToRow(row: Row): Row ={
      Row.fromSeq(row(0).asInstanceOf[DenseVector].toArray ++ Array(row(1).toString))
    }

    val schemas = new StructType(df.columns.reverse.tail.map(x=>StructField(x, DoubleType, nullable=false)).reverse
      ++ Array(StructField(df.columns.reverse.head, StringType, nullable = false)))

    val rows = sampledData.collect().map(x=>vecToRow(x))
    val d: RDD[Row] = sampledData.sparkSession.sparkContext.parallelize(rows)

    val result = sampledData.sparkSession.createDataFrame(d, schemas)

    result.repartition(1).
      write.format("com.databricks.spark.csv").
      option("header", "true").
      mode("overwrite").
      save(savePath)
  }
}

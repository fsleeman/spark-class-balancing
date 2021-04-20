package org.apache.spark.ml.sampling

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{MinMaxScaler, StringIndexer}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.io.Source
import scala.util.Random
import org.apache.spark.ml.sampling.utilities.{convertFeaturesToVector, getCountsByClass}

import org.apache.spark.mllib.linalg.{Vector, Vectors}

//FIXME - turn classes back to Ints instead of Doubles
object RunExample {




  def main(args: Array[String]) {

    val filename = "/home/ford/data/oversample.txt"
    val lines = Source.fromFile(filename).getLines.map(x=>x.split(":")(0)->x.split(":")(1)).toMap
    val input_file = lines("dataset").trim
    // val classifier = lines("classifier").trim
    val samplingMethods = lines("sampling").split(",").map(x=>x.trim)
    // val sampling = lines("sampling").trim
    // val labelColumnName = lines("labelColumn").trim
    val enableDataScaling = if(lines("enableScaling").trim == "true") true else false
    // val numSplits = lines("numCrossValidationSplits").trim.toInt
    val savePath = lines("savePath").trim

    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()
    val df = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv(input_file) //.withColumnRenamed(labelColumnName, "label")

     import spark.implicits._

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString//.toDouble.toInt
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    results.show()
    results.printSchema()

   val converted: DataFrame = convertFeaturesToVector(results)
    converted.show()
    converted.printSchema()

    val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)


    val scaledData: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.show()
      scaledData.printSchema()

      scaledData.drop("features").withColumn("features", toDense($"scaledFeatures")).drop("scaledFeatures")
    } else { converted }.cache()

    scaledData.show()
    scaledData.printSchema()

    // val columnToString = udf((d: Double) => d.toString)

    getCountsByClass("label", scaledData).show
    scaledData.show()
    scaledData.printSchema()

    val samplingMapX: Map[Double, Double] =
      Map(
        3.0 -> 2.0,
        4.0 -> 2.0
      )
    val samplingMap: Map[Double, Double] =
      Map(
        1.0 -> 2.0,
        2.0 -> 2.0
      )
    // FIXME - put sampling methods inside a class

    for(samplingMethod <- samplingMethods) {
      val trainData = scaledData
      println("$$$$$$$$$$$$$$$$$$$" + samplingMethod + "$$$$$$$$$$$$$$$$$$$")
      val t0 = System.nanoTime()
      val oversampled = if (samplingMethod == "kMeansSmote") {
        val r = new KMeansSMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true) //.setTopTreeSize(2)
        model.transform(trainData)
      } else if (samplingMethod == "borderlineSmote") {
        val r = new BorderlineSMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "rbo") {
        val r = new RBO().setGamma(0.001).setStepSize(0.001).setIterations(100).setStoppingProbability(0.001)
        val model = r.fit(trainData).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "adasyn") {
        val r = new ADASYN().setBalanceThreshold(0.0)
        val model = r.fit(trainData).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "safeLevel") {
        val r = new SafeLevelSMOTE()
        val model = r.fit(trainData).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "smote") {
        val r = new SMOTE
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true) //.setTopTreeSize(2)   //.setSamplingRatios(samplingMap)
        model.transform(trainData)
      } else if (samplingMethod == "mwmote") {
        val r = new MWMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true) //.setTopTreeSize(2)
        model.transform(trainData)
      } else if (samplingMethod == "ccr") {
        val r = new CCR().setEnergy(1.0)
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true) // .setTopTreeSize(10)
        model.transform(trainData)
      } else if (samplingMethod == "ans") {
        val r = new ANS().setdDstanceNeighborLimit(100)
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "clusterSmote") {
        val r = new ClusterSMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true) //.setTopTreeSize(2)
        model.transform(trainData)
      } else if (samplingMethod == "gaussianSmote") {
        val r = new GaussianSMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "smote_d") {
        val r = new SMOTED()
        val model = r.fit(trainData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "nras") {
        val r = new NRAS()
        val model = r.fit(trainData).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "randomOversample") {
        val r = new RandomOversample() // multi-class done
        val model = r.fit(trainData).setSamplingRatios(samplingMap).setOversamplesOnly(true)
        model.transform(trainData)
      } else if (samplingMethod == "randomUndersample") {
        val r = new RandomUndersample() // multi-class done
        val model = r.fit(trainData)
        model.transform(trainData)
      } else {
        trainData
      }



      //val r = new SMOTE
      //val model = r.fit(scaledData).setBalanceThreshold(0.0).setSamplingRatios(samplingMap2)
      //val oversampled = model.transform(scaledData)

      //val r = new RandomOversample()
      //val model = r.fit(scaledData).setSamplingRatios(samplingMap2)
      //val oversampled = model.transform(scaledData)
      oversampled.show
      getCountsByClass("label", oversampled).show

      def vecToArray(row: Row): Array[Double] = {
        row(0).asInstanceOf[DenseVector].toArray ++ Array(row(1).toString.toDouble) //.asInstanceOf[Double])
      }

      val collected = oversampled.collect().map(x => vecToArray(x))
      val rows: Array[Row] = collected.map { x => Row(x: _*) }
      println(rows(0))

      import org.apache.spark.sql.types._
      val d: RDD[Row] = oversampled.sparkSession.sparkContext.parallelize(rows)
      val schema = new StructType(Array(
        StructField("0", DoubleType, nullable = false),
        StructField("1", DoubleType, nullable = false),
        StructField("label", DoubleType, nullable = false)))

      val result = oversampled.sparkSession.createDataFrame(d, schema)
      result.show

      println("save path: " + savePath)
      result.repartition(1).
        write.format("com.databricks.spark.csv").
        option("header", "true").
        mode("overwrite").
        save(savePath + "/" + samplingMethod)
    }

  }

}
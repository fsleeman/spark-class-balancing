package org.apache.spark.ml.sampling

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest, RandomForestClassifier}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vectors}
import org.apache.spark.ml.sampling.Utilities.{convertFeaturesToVector, getCountsByClass}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.io.Source
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.ml.classification.{LinearSVC, OneVsRest, RandomForestClassifier}
import org.apache.spark.ml.sampling.SamplingExperiments.filterByMinorityType


//FIXME - turn classes back to Ints instead of Doubles
object Hyperparam {

  def getStratifiedSplit(dfs: Array[DataFrame], totalSplits: Int, splitIndex: Int): (DataFrame, DataFrame) ={
    val splitIndexLow = (splitIndex)/ totalSplits.toDouble
    val splitIndexHigh = (splitIndex + 1)/ totalSplits.toDouble
    val testFiltered = dfs.map(x => x.filter(x("index") < x.count() * splitIndexHigh && x("index") >= x.count() * splitIndexLow)) // .drop("index"))
    val testDF = testFiltered.reduce(_ union _)

    val trainFiltered = dfs.map(x=>x.filter(x("index") >= x.count() * splitIndexHigh || x("index") < x.count() * splitIndexLow))
    val trainDF=  trainFiltered.reduce(_ union _)

    println("train")
    getCountsByClass("label", trainDF).show()
    println("test")
    getCountsByClass("label", testDF).show()


    (trainDF, testDF)
  }


  def main(args: Array[String]) {

    val filename = args(0)// "/home/ford/data/sampling_input.txt"
    val lines = Source.fromFile(filename).getLines.map(x=>x.split(":")(0)->x.split(":")(1)).toMap
    // val input_file = lines("dataset").trim
    val classifier: String = lines("classifier").trim
    val samplingMethods = lines("sampling").split(",").map(x=>x.trim)
    val labelColumnName = lines("labelColumn").trim
    val enableDataScaling = if(lines("enableScaling").trim == "true") true else false
    val numSplits = lines("numCrossValidationSplits").trim.toInt
    // val minorityTypePath = lines("minorityTypePath").trim
    val savePath = lines("savePath").trim
    val datasetSize = lines("datasetSize").trim
    val datasetPath = lines("datasetPath").trim
    val datasetName = lines("datasetName").trim
    val maxTrees = lines("maxTrees").trim.split(",").map(x=>x.toInt)
    val maxDepth = lines("maxDepth").trim.split(",").map(x=>x.toInt)
    val maxIter = lines("maxTrees").trim.split(",").map(x=>x.toInt)
    val regParam = lines("maxDepth").trim.split(",").map(x=>x.toDouble)


    val input_file = datasetPath + "/" + datasetName  + datasetSize + ".csv"
    println("input")
    for(x<-lines) {
      println(x)
    }
    println("Dataset: " + input_file)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val df = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv(input_file).withColumnRenamed(labelColumnName, "label")

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    // val results = indexDF(df)

    results.show()
    results.printSchema()



    val converted: DataFrame = convertFeaturesToVector(results)

    val asDense = udf((v: Any) => {
      if(v.isInstanceOf[SparseVector]) {
        v.asInstanceOf[SparseVector].toDense
      } else {
        v.asInstanceOf[DenseVector]
      }
    })

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndexed")

    val datasetIndexed = indexer.fit(converted).transform(converted).drop("label")
      .withColumnRenamed("labelIndexed", "label")

    println("here")
    datasetIndexed.show()
    datasetIndexed.printSchema()

    val scaledData = datasetIndexed

    scaledData.show()

    val countsBy = getCountsByClass("label", scaledData)
    val labelList = countsBy.select("_1").collect().map(x=>x(0).toString.toDouble)
    val clsFiltered = labelList.map(x=>scaledData.filter(scaledData("label")===x))

    val dfs: (DataFrame, DataFrame) = getStratifiedSplit(clsFiltered, 5, 0)
    val trainData = dfs._1
    val testData = dfs._2

    val cls = if(classifier == "svm") {
      val svm = new LinearSVC()
        .setLabelCol("label")
        .setFeaturesCol("features")

      val ovr = new OneVsRest().setClassifier(svm)

      val paramGrid = new ParamGridBuilder()
        .addGrid(svm.maxIter, maxIter)
        .addGrid(svm.regParam, regParam)
        .build()

      (ovr, paramGrid)
    } else {
      val rf = new RandomForestClassifier()
        .setLabelCol("label")
        .setFeaturesCol("features")

      val paramGrid = new ParamGridBuilder()
        .addGrid(rf.numTrees, maxTrees) // 100
        .addGrid(rf.maxDepth, maxDepth)
        .build()

      (rf, paramGrid)
    }

    val algorithm = cls._1
    val paramGrid = cls._2

    val cv = new CrossValidator()
      .setEstimator(algorithm)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(4)  // Evaluate up to 2 parameter settings in parallel

    val cvModel = cv.fit(trainData)

    cvModel.transform(testData)
    println(cvModel.bestModel)

    // println(cvModel.getEstimatorParamMaps)
    println("*********")
    println(cvModel.bestModel.extractParamMap())
    /*for(x<-cvModel.bestModel.explainParams()) {
      print(x.toString())
    }*/
      /*.select("index", "text", "probability", "prediction")
      .collect()
      .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
        println(s"($id, $text) --> prob=$prob, prediction=$prediction")
      }*/
  }

}

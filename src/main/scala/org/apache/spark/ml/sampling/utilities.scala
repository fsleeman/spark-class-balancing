package org.apache.spark.ml.sampling

import org.apache.spark.ml.Model
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions.{count, udf}
import org.apache.spark.sql.types._
import scala.util.Random


object utilities {

  /**
    * Trait for shared param featuresCol (default: "features").
    */
  trait HasFeaturesCol extends Params {

    /**
      * Param for features column name.
      * @group param
      */
    final val featuresCol: Param[String] = new Param[String](this, "featuresCol", "features column name")

    setDefault(featuresCol, "features")

    /** @group getParam */
    final def getFeaturesCol: String = $(featuresCol)
  }

  /**
    * Trait for shared param labelCol (default: "label").
    */
  trait HasLabelCol extends Params {

    /**
      * Param for label column name.
      * @group param
      */
    final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")

    setDefault(labelCol, "label")

    /** @group getParam */
    final def getLabelCol: String = $(labelCol)
  }

  trait ClassBalancingRatios extends Params {

    /**
      * Param map for setting class balancing ratios
      * Default: Empty map
      *
      * @group param
      */
    // val samplingRatios = new Param[Map[Double, Double]](this, "samplingRatios", "map of sampling ratios per class")
    final val samplingRatios = new Param[Map[Double, Double]](this, "samplingRatios", "map of sampling ratios per class")


    /** @group getParam */
    def getSamplingRatios: Map[Double, Double] = $(samplingRatios)

    final def setSamplingRatios(value: Map[Double, Double]): this.type = set(samplingRatios, value)

    /**
      * Param map for specifying if only oversampled synthetic examples be returned or original examples as well
      * Default: false
      *
      * @group param
      */
    final val oversamplesOnly = new Param[Boolean](this, "oversamplesOnly", "should only oversampled synthetic examples be returned")

    /** @group getParam */
    def getOversamplesOnly: Boolean = $(oversamplesOnly)

    final def setOversamplesOnly(value: Boolean): this.type = set(oversamplesOnly, value)

    setDefault(samplingRatios -> Map(), oversamplesOnly -> false)
  }

  trait UsingKNN extends Params {

    /**
      * Param for kNN k-value.
      * @group param
      */
    final val k: Param[Int] = new Param[Int](this, "k", "k-value for kNN")

    /** @group getParam */
    final def setK(value: Int): this.type = set(k, value)

    /**
      * Param for kNN top tree size.
      * @group param
      */
    final val topTreeSize: Param[Int] = new Param[Int](this, "topTreeSize", "Size of the top tree for the kNN tree structure")

    /** @group getParam */
    final def setTopTreeSize(value: Int): this.type = set(topTreeSize, value)

    /**
      * Param for kNN top tree leaf size.
      * @group param
      */
    final val topTreeLeafSize: Param[Int] = new Param[Int](this, "topTreeLeafSize", "Size of the top tree leaves for the kNN tree structure")

    /** @group getParam */
    final def setTopTreeLeafSize(value: Int): this.type = set(topTreeLeafSize, value)

    /**
      * Param for kNN top tree leaf size.
      * @group param
      */
    final val subTreeLeafSize: Param[Int] = new Param[Int](this, "subTreeLeafSize", "Size of the top sub tree leaves for the kNN tree structure")

    /** @group getParam */
    final def setSubTreeLeafSize(value: Int): this.type = set(subTreeLeafSize, value)

    /**
      * Param for kNN top tree leaf size.
      * @group param
      */
    final val balanceThreshold: Param[Double] = new Param[Double](this, "balanceThreshold", "Balance threshold for determining tree splits")

    /** @group getParam */
    final def setBalanceThreshold(value: Double): this.type = set(balanceThreshold, value)

    setDefault(k -> 5, topTreeSize -> 10, topTreeLeafSize -> 100, subTreeLeafSize -> 100, balanceThreshold -> 0.7)
  }


  def calculateToTreeSize(topTreeSize: Int, datasetCount: Long): Int ={
    if(topTreeSize >= datasetCount.toInt || datasetCount.toInt < 100) {
      if(datasetCount.toInt == 0) {
        println("~~~~~~~~~~~~~~~~~~ zero datasetCount")
      } else {
        println("~~~~~~~~~~~~~~~~~~ datasetCount " + datasetCount.toInt)
      }

      datasetCount.toInt
    } else {
      topTreeSize
    }
  }

  def getSamplesToAdd(label: Double, sampleCount: Long, majorityClassCount: Int, samplingRatios: Map[Double, Double]): Int ={
    println("%% samplesToAdd " + label + " " + sampleCount + " " + majorityClassCount)
    println(samplingRatios)

    if(samplingRatios contains label) {
      val ratio = samplingRatios(label)
      if(ratio <= 1) {
        println("sample count 0")
        0
      } else {
        println("sample count " + ((ratio - 1.0) * sampleCount).toInt)
        ((ratio - 1.0) * sampleCount).toInt
      }
    } else {
      println("sample count " + (majorityClassCount - sampleCount.toInt))
      majorityClassCount - sampleCount.toInt
    }
  }

  def getCountsByClass(label: String, df: DataFrame): DataFrame = {
    val numberOfClasses = df.select("label").distinct().count()
    val aggregatedCounts = df.groupBy(label).agg(count(label)).take(numberOfClasses.toInt) //FIXME

    val sc = df.sparkSession.sparkContext
    //val countSeq = aggregatedCounts.map(x => (x(0).toString, x(1).toString.toInt)).toSeq
    val countSeq = aggregatedCounts.map(x => (x(0).toString, x(1).toString)).toSeq
    val rdd = sc.parallelize(countSeq)

    df.sparkSession.createDataFrame(rdd)
  }

  def convertFeaturesToVector(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._
    val convertToVector = udf((array: Seq[Double]) => {
      Vectors.dense(array.map(_.toDouble).toArray)
    })

    df.withColumn("features", convertToVector($"features"))
  }

  val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)

  def createSmoteStyleExample(features: DenseVector, randomNeighbor: DenseVector): DenseVector ={
    val gap = Random.nextDouble()
    Vectors.dense(Array(features.toArray, randomNeighbor.toArray).transpose.map(x=>x(0) + gap * (x(1) - x(0)))).toDense
  }

}

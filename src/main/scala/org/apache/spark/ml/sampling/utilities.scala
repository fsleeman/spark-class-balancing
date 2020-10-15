package org.apache.spark.ml.sampling

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{count, udf}
import org.apache.spark.sql.types._

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
    val samplingRatios = new Param[Map[Double, Double]](this, "samplingRatios", "map of sampling ratios per class")

    setDefault(samplingRatios -> Map())

    /** @group getParam */
    def getSamplingRatios: Map[Double, Double] = $(samplingRatios)

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

  def getSamplesToAdd(label: Double, sampleCount: Long, majorityClassCount: Int, samplingRatios: Map[Double, Double]): Int ={
    println("samplesToAdd " + sampleCount + " " + majorityClassCount)
    println(samplingRatios)

    if(samplingRatios contains label) {
      val ratio = samplingRatios(label)
      if(ratio <= 1) {
        0
      } else {
        ((ratio - 1.0) * sampleCount).toInt
      }
    } else {
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

}

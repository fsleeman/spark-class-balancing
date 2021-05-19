package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, getSamplesToAdd, getSamplingMap}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.util.Random


/** Transformer Parameters*/
private[ml] trait RBOModelParams extends Params with HasFeaturesCol with HasLabelCol with ClassBalancingRatios {

  /**
    * Param for gamma
    * @group param
    */
  final val gamma: Param[Double] = new Param[Double](this, "gamma", "gamma value")

  /** @group getParam */
  final def setGamma(value: Double): this.type = set(gamma, value)

  /**
    * Param for iterations
    * @group param
    */
  final val iterations: Param[Int] = new Param[Int](this, "iterations", "iterations per synthetic example")

  /** @group getParam */
  final def setIterations(value: Int): this.type = set(iterations, value)

  /**
    * Param for stepSize
    * @group param
    */
  final val stepSize: Param[Double] = new Param[Double](this, "stepSize", "stepSize")

  /** @group getParam */
  final def setStepSize(value: Double): this.type = set(stepSize, value)

  /**
    * Param for stoppingProbability
    * @group param
    */
  final val stoppingProbability: Param[Double] = new Param[Double](this, "stoppingProbability", "probability of stopping early")

  /** @group getParam */
  final def setStoppingProbability(value: Double): this.type = set(stoppingProbability, value)

  setDefault(gamma -> 1.0, iterations -> 1, stepSize -> 0.01, stoppingProbability -> 1.0)
}

/** Transformer */
class RBOModel private[ml](override val uid: String) extends Model[RBOModel] with RBOModelParams {
  def this() = this(Identifiable.randomUID("rbo"))

  def pointDifference(x1: Array[Double], x2: Array[Double]): Double = {
    val combined = Array[Array[Double]](x1, x2)
    val difference: Array[Double] = combined.transpose.map(x=>Math.abs(x(0)-x(1)))
    difference.sum
  }

  def calculatePhi(spark: SparkSession, x: Array[Double], K: Array[Array[Double]], k: Array[Array[Double]], gamma: Double): Double = {
    val majorityValue = spark.sparkContext.parallelize(K).map(Ki=>Math.exp(-Math.pow(pointDifference(Ki, x)/gamma, 2))).collect().sum /// FIXME - Super slow
    val minorityValue = spark.sparkContext.parallelize(k).map(ki=>Math.exp(-Math.pow(pointDifference(ki, x)/gamma, 2))).collect().sum /// FIXME - Super slow

    majorityValue - minorityValue
  }

  def getRandomStopNumber(numIterations: Int, p: Double) : Int = {
    if (p == 1.0) {
      numIterations
    } else {
      val iterationCount = numIterations * p + (Random.nextGaussian() * numIterations * p).toInt
      if (iterationCount > 1) {
        iterationCount.toInt
      } else {
        1
      }
    }
  }

  def createExample(spark: SparkSession, majorityExamples: Array[Array[Double]],
                    minorityExamples: Array[Array[Double]], gamma: Double, stepSize: Double, numInterations: Int,
                    p: Double): DenseVector ={
    val featureLength = minorityExamples(0).length
    var point = minorityExamples(Random.nextInt(minorityExamples.length))

    val pointPhi = calculatePhi(spark, point, majorityExamples, minorityExamples, gamma)
    for(_ <- 0 until getRandomStopNumber(numInterations, p)) {
      val directions = Set(-1, 1)
      val d: Array[Double] = (0 until featureLength).map(_=>directions.toVector(Random.nextInt(directions.size)).toDouble).toArray
      val v: Array[Double] = (0 until featureLength).map(_=>Random.nextDouble()).toArray

      val translated: Array[Double] = Array(point, v, d).transpose.map(x=>x(0) + x(1) * x(2) * stepSize)
      if(calculatePhi(spark, translated, majorityExamples, minorityExamples, gamma) < pointPhi) {
        point = translated
      }
    }
    Vectors.dense(point).toDense
  }

  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val spark = dataset.sparkSession

    val minorityDF = dataset.filter(dataset($(labelCol)) === minorityClassLabel)
    val majorityDF = dataset.filter(dataset($(labelCol)) =!= minorityClassLabel)

    val minorityExamples = minorityDF.select($(featuresCol)).collect.map(x=>x(0).asInstanceOf[DenseVector].toArray)
    val majorityExamples = majorityDF.select($(featuresCol)).collect.map(x=>x(0).asInstanceOf[DenseVector].toArray)

    val addedPoints = (0 until samplesToAdd).map(_=>createExample(spark, majorityExamples,
      minorityExamples, $(gamma), $(stepSize), $(iterations), $(stoppingProbability)))

    val sampledArray: Array[(Double, DenseVector)] = addedPoints.map(x=>(minorityClassLabel, x)).toArray
    val sampledDF = spark.createDataFrame(spark.sparkContext.parallelize(sampledArray))
    sampledDF.withColumnRenamed("_1", "label")
      .withColumnRenamed("_2", "features")
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    val indexer = new StringIndexer()
      .setInputCol($(labelCol))
      .setOutputCol("labelIndexed")

    val datasetIndexed = indexer.fit(dataset).transform(dataset)
      .withColumnRenamed($(labelCol), "originalLabel")
      .withColumnRenamed("labelIndexed",  $(labelCol))
    datasetIndexed.show()
    datasetIndexed.printSchema()

    val labelMap = datasetIndexed.select("originalLabel",  $(labelCol)).distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
    val labelMapReversed = labelMap.map(x=>(x._2, x._1))

    val datasetSelected = datasetIndexed.select($(labelCol), $(featuresCol))
    val counts = getCountsByClass(datasetSelected.sparkSession, $(labelCol), datasetSelected.toDF).sort("_2")
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplingMapConverted: Map[Double, Double] = getSamplingMap($(samplingRatios), labelMap)
    val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)
    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected, x))
      .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble,
        datasetSelected.filter(datasetSelected($(labelCol))===clsList(x._3)).count(),
        majorityClassCount, samplingMapConverted)))

    val balancedDF = if($(oversamplesOnly)) {
      clsDFs.reduce(_ union _)
    } else {
      datasetIndexed.select($(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    }

    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balancedDF.withColumn("originalLabel", restoreLabel(balancedDF.col($(labelCol)))).drop($(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))

  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): RBOModel = {
    val copied = new RBOModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait RBOParams extends RBOModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class RBO(override val uid: String) extends Estimator[RBOModel] with RBOParams {
  def this() = this(Identifiable.randomUID("rbo"))

  override def fit(dataset: Dataset[_]): RBOModel = {
    val model = new RBOModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): RBO = defaultCopy(extra)

}

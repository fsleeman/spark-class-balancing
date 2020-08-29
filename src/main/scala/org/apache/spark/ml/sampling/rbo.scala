package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random




/** Transformer Parameters*/
private[ml] trait RBOModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class RBOModel private[ml](override val uid: String) extends Model[RBOModel] with RBOModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  def pointDifference(x1: Array[Double], x2: Array[Double]): Double = {
    val combined = Array[Array[Double]](x1, x2)
    val difference: Array[Double] = combined.transpose.map(x=>Math.abs(x(0)-x(1)))
    difference.sum
  }

  def calculatePhi(x: Array[Double], K: Array[Array[Double]], k: Array[Array[Double]], gamma: Double): Double = {
    val majorityValue = K.map(Ki=>Math.exp(-Math.pow(pointDifference(Ki, x)/gamma, 2)))
    val minorityValue = k.map(ki=>Math.exp(-Math.pow(pointDifference(ki, x)/gamma, 2)))

    //println(majorityValue.sum, minorityValue.sum)
    majorityValue.sum - minorityValue.sum
  }

  def getRandomStopNumber(numInterations: Int, p: Double) : Int = {
    if (p == 1.0) {
      numInterations
    } else {
      val iterationCount = numInterations * p + (Random.nextGaussian() * numInterations * p).toInt
      if (iterationCount > 1) {
        iterationCount.toInt
      } else {
        1
      }
    }
  }

  def createExample(majorityExamples: Array[Array[Double]], minorityExamples: Array[Array[Double]], gamma: Double, stepSize: Double, numInterations: Int, p: Double): DenseVector = {
    println("****** at create Example *****")
    println("numIterations: " + numInterations)
    println("stopIndex: " + getRandomStopNumber(numInterations, p))

    val featureLength = minorityExamples(0).size
    println("feature length: " + featureLength)
    var point = minorityExamples(Random.nextInt(minorityExamples.length))

    for(_<-0 until getRandomStopNumber(numInterations, p)) {
      val directions = Set(-1, 1)
      val d: Array[Double] = (0 until featureLength).map(x=>directions.toVector(Random.nextInt(directions.size)).toDouble).toArray
      val v: Array[Double] = (0 until featureLength).map(x=>Random.nextDouble()).toArray

      val translated: Array[Double] = Array(point, v, d).transpose.map(x=>x(0) + x(1) * x(2) * stepSize)
      if(calculatePhi(translated, majorityExamples, minorityExamples, gamma) < calculatePhi(point, majorityExamples, minorityExamples, gamma)) {
        point = translated
      }
    }
    Vectors.dense(point).toDense
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF
    val spark = df.sparkSession
    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    println(minClassLabel, minClassCount)
    println(maxClassLabel, maxClassCount)

    val minorityDF = df.filter(df("label") === minClassLabel)
    val majorityDF = df.filter(df("label") =!= minClassLabel)

    val gamma = 1
    val numInterations = 1
    val stepSize = 0.01
    val p = 1.0

    println("majority " + majorityDF.count)
    majorityDF.show
    println("minority " + minorityDF.count)
    minorityDF.show

    val minorityExamples = minorityDF.collect.map(row=>row.getValuesMap[Any](row.schema.fieldNames)("features")).map(x=>x.asInstanceOf[DenseVector].toArray)
    val majorityExamples = majorityDF.collect.map(row=>row.getValuesMap[Any](row.schema.fieldNames)("features")).map(x=>x.asInstanceOf[DenseVector].toArray)

    val addedPoints = (0 until 10).map(_=>createExample(majorityExamples, minorityExamples, gamma, stepSize, numInterations, p))

    val foo: Array[(Long, Int, DenseVector)] = addedPoints.map(x=>(0.toLong, minClassLabel.toInt, x)).toArray
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))
    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")
    val all = df.union(bar2)

    all
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
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): RBOModel = {
    val model = new RBOModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): RBO = defaultCopy(extra)

}

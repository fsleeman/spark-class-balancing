package org.apache.spark.ml.sampling

import org.apache.spark.ml._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, VectorUDT, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{SchemaUtils, _}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, desc, udf}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.sampling.Sampling._

import org.apache.spark.mllib.linalg.MatrixImplicits._
import org.apache.spark.mllib.linalg.VectorImplicits._

import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, DenseVector => OldDenseVector,
  Matrices => OldMatrices, Vector => OldVector, Vectors => OldVectors}

import scala.util.Random

/*private [sampling] trait KMeansSMOTE extends Params with HasInputCol with HasOutputCol{
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }
}

private [sampling] trait KMeansSMOTEModelParams extends KMeansSMOTE {

}*/



/*
class KMeansSMOTE (override val uid: String)
extends Estimator[KMeansSMOTEModel] with KMeansSMOTE with DefaultParamsWritable {

  override def fit(dataset: Dataset[_]): KMeansSMOTEModelParams = {
    transformSchema(dataset.schema, logging = true)
    val input: RDD[OldVector] = dataset.select($(inputCol)).rdd.map {
      case Row(v: Vector) => OldVectors.fromML(v)
    }
    val pca = new sampling.KMeansSMOTE() // k = $(k)
    val pcaModel = pca.fit(input)
    copyValues(new KMeansSMOTEModel().setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}*/







/** Transformer Parameters*/
private[ml] trait KMeansSMOTEModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class KMeansSMOTEModel private[ml](override val uid: String) extends Model[KMeansSMOTEModel] with KMeansSMOTEModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  private def getFeaturePoint(ax: Double, bx: Double) : Double ={
    Random.nextDouble() * (maxValue(ax, bx) - minValue(ax, bx)) + minValue(ax, bx)
  }

  private def getSmotePoint(a: Array[Double], b: Array[Double]): Array[Double] = {
    a.indices.map(x => getFeaturePoint(a(x), b(x))).toArray
  }

  private def getSmoteExample(knnExamples: Array[Array[Double]]): Array[Double] ={
    val a = knnExamples(0)
    val b = knnExamples(Random.nextInt(knnExamples.length))
    getSmotePoint(a, b)
  }

  /// FIXME - check parameters
  private def sampleCluster(df: DataFrame, cls: Int, samplesToAdd: Int): Array[Row] = {

    val leafSize = 100
    val kValue = 5
    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(kValue + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f: KNNModel = model.fit(df)
    val t = f.transform(df)
    t.show

    val collected = t.collect()
    val count = df.count.toInt

    // examples/k neighbors/features of Doubles
    val examples: Array[Array[Array[Double]]] = collected.map(x=>x(4).toString.substring(13, x(4).toString.length - 3).split("]], ").map(x=>x.split(""",\[""")(1)).map(y=>y.split(",").map(z=>z.toDouble)))
    val randomIndicies = (0 to samplesToAdd).map(_ => Random.nextInt(count))

    val syntheticExamples = randomIndicies.par.map(x=>getSmoteExample(examples(x))).toArray

    syntheticExamples.map(x=>Row(0, cls, Vectors.dense(x)))
  }

  val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
  val toArrUdf: UserDefinedFunction = udf(toArr)

  private def getTotalElementDistance(current: Array[Double], rows: Array[Array[Double]]): Double ={
    rows.map(x=>getSingleDistance(x, current)).sum
  }

  private def getSparsity(data: Dataset[Row], imbalancedRatio: Double): Double = {
    // calculate all distances for minority class
    println("at sparsitiy count " + data.count())
    // data.show()
    data.printSchema()
    val collected: Array[Array[Double]] =  data.withColumn("features", toArrUdf(col("features"))).select("features").collect().map(x=>x.toString.substring(14, x.toString.length-2).split(",").map(x=>x.toDouble))
    val n = collected.length // number of minority examples in cluster
    val m = collected(0).length // number of features
    val meanDistance = collected.map(x=>getTotalElementDistance(x, collected)).sum / ((n * n) - n)
    val density = n / Math.pow(meanDistance, m)
    val sparsity = 1 / density
    sparsity
  }

  private def getSingleDistance(x: Array[Double], y: Array[Double]): Double = {
    var distance = 0.0
    for(index<-x.indices) {
      distance += (x(index) -  y(index)) *(x(index) - y(index))
    }
    distance
  }

  private def getImbalancedRatio(spark: SparkSession, data: Dataset[Row], minClassLabel: String): Double = {
    val minorityCount = data.filter(data("label") === minClassLabel).count
    val majorityCount = data.filter(data("label") =!= minClassLabel).count
    (minorityCount + 1) / (majorityCount + 1).toDouble
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val k = 5 // FIXME
    val imbalanceRatioThreshold = 1.0 // FIXME - make parameter
    val kSmote = 5          // FIXME - make parameter
    // val densityExponent = 10 // FIXME - number of features
    // cluster

    val df = dataset.filter((dataset("label") === 1) || (dataset("label") === 5)).toDF // FIXME
    val spark = df.sparkSession
    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    val kmeans = new KMeans().setK(k).setSeed(1L) // FIXME - fix seed
    val model = kmeans.fit(df)
    val predictions = model.transform(df)

    // filter
    val clusters = (0 until k).map(x=>predictions.filter(predictions("prediction")===x)).toArray

    val imbalancedRatios = clusters.map(x=>getImbalancedRatio(spark, x, minClassLabel))

    val sparsity = (0 until k).map(x=>getSparsity(predictions.filter((predictions("prediction")===x)
      && (predictions("label")===minClassLabel)), imbalancedRatios(x)))
    val sparsitySum = sparsity.sum

    val classSparsity = (0 until k).map(x=>(x, ((sparsity(x)/sparsitySum) * samplesToAdd).toInt))

    var sampledDataset: Array[Row] = Array[Row]()

    for(x<-classSparsity) {
      println(x._1, x._2)
      if(x._2 > 0) {
        sampledDataset = sampledDataset ++ sampleCluster(predictions.filter(predictions("prediction")===x._1 && predictions("label")===minClassLabel), x._1, x._2)
      }
    }

    val foo: Array[(Long, Int, DenseVector)] = sampledDataset.map(x=>x.toSeq).map(x=>(x(0).toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector]))

    println("sampled count: " + sampledDataset.length)
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))
    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    bar2.show
    println(bar2.count)

    val all = df.union(bar2)
    all.show()
    getCountsByClass(spark, "label", df).show
    println("all: " + all.count())

    // over sampling
    all
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): KMeansSMOTEModel = {
    val copied = new KMeansSMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}




/** Estimator Parameters*/
private[ml] trait KMeansSMOTEParams extends KMeansSMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class KMeansSMOTE(override val uid: String) extends Estimator[KMeansSMOTEModel] with KMeansSMOTEParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): KMeansSMOTEModel = {
    val model = new KMeansSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): KMeansSMOTE = defaultCopy(extra)

}

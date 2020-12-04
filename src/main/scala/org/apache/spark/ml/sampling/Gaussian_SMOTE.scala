package org.apache.spark.ml.sampling

import breeze.stats.distributions.Gaussian
import org.apache.hadoop.hdfs.server.datanode.fsdataset.impl.RamDiskReplicaLruTracker
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, calculateToTreeSize, getSamplesToAdd}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.sql.expressions.UserDefinedFunction

import scala.collection.mutable
import scala.util.Random


/** Transformer Parameters*/
private[ml] trait GaussianSMOTEModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {

  /**
    * Param for gaussian sigma.
    * @group param
    */
  final val sigma: Param[Double] = new Param[Double](this, "sigma", "gaussian sigma")

  /** @group getParam */
  final def setSigma(value: Double): this.type = set(sigma, value)

  setDefault(sigma -> 0.5)
}

/** Transformer */
class GaussianSMOTEModel private[ml](override val uid: String) extends Model[GaussianSMOTEModel] with GaussianSMOTEModelParams {
  def this() = this(Identifiable.randomUID("gaussianSmote"))

  def getSingleDistance(x: Array[Double], y: Array[Double]): Double = {
    var distance = 0.0
    for(index<-x.indices) {
      distance += (x(index) -  y(index)) *(x(index) - y(index))
    }
    distance
  }

  def getGaussian(u: Double, s: Double): Double = {
    val g = Gaussian(u, s)
    g.draw()
  }

  def getSmoteSample(row: Row): Row = {
    val label = row(0).toString.toDouble
    val features = row(1).asInstanceOf[DenseVector].toArray
    val neighbors = row(2).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail // skip the self neighbor
    val randomNeighbor = neighbors(Random.nextInt(neighbors.length)).toArray

    val mean = Random.nextDouble()
    val range = getGaussian(mean, $(sigma))
    val syntheticExample = Vectors.dense(Array(features, randomNeighbor).transpose.map(x => x(0) + (x(1) - x(0)) * range)).toDense

    Row(label, syntheticExample)
  }

  // FIXME - move these to utilities
  val checkForNegatives: UserDefinedFunction = udf((features: DenseVector) => {
    if(features.values.min < 0.0 || features.values.count(_.isNaN) > 0) {
      true
    } else {
      false
    }
  })

  def removeNegatives(df: DataFrame): DataFrame ={
    val negatives = df.withColumn("negativesPresent", checkForNegatives(df.col($(featuresCol))))
    negatives.filter(negatives("negativesPresent")=!=true).drop("negativesPresent")
  }

  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val df = dataset.toDF
    val spark = df.sparkSession
    import spark.implicits._

    val minorityDF = df.filter(df($(labelCol)) === minorityClassLabel)

    /*** For each minority example, calculate the m nn's in training set***/
    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), minorityDF.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel = model.fit(minorityDF)
    val minorityDataNeighbors = fitModel.transform(minorityDF)

    val minorityDataNeighborsCount = minorityDF.count.toInt

    val rnd = uniformRDD(spark.sparkContext, samplesToAdd).map(x=>Math.floor(x * minorityDataNeighborsCount).toInt).collect()
    val collected = minorityDataNeighbors.withColumn("neighborFeatures", $"neighbors.features").drop("neighbors").collect
    val createdSamples = spark.createDataFrame(spark.sparkContext.parallelize(rnd.map(x=>getSmoteSample(collected(x)))), dataset.schema)

    removeNegatives(createdSamples)
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

    val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)

    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected.filter(datasetSelected($(labelCol))===clsList(x))))
      .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, x._2.count, majorityClassCount, $(samplingRatios))))

    val balanecedDF = datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))//.repartition(1)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): GaussianSMOTEModel = {
    val copied = new GaussianSMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}

/** Estimator Parameters*/
private[ml] trait GaussianSMOTEParams extends GaussianSMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class GaussianSMOTE(override val uid: String) extends Estimator[GaussianSMOTEModel] with GaussianSMOTEParams {
  def this() = this(Identifiable.randomUID("gaussianSmote"))

  override def fit(dataset: Dataset[_]): GaussianSMOTEModel = {
    val model = new GaussianSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): GaussianSMOTE = defaultCopy(extra)

}

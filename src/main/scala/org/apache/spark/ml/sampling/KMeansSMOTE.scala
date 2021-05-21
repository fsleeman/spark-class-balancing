package org.apache.spark.ml.sampling

import org.apache.spark.ml._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, desc, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, getSamplesToAdd, getSamplingMap}
import org.apache.spark.ml.sampling.utils.getCountsByClass

import scala.util.Random

/** Transformer Parameters*/
private[ml] trait KMeansSMOTEModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {
  /**
    * Param for kNN k-value.
    * @group param
    */
  final val clusterK: Param[Int] = new Param[Int](this, "clusterK", "cluster k-value for kNN")

  /** @group getParam */
  final def setClusterK(value: Int): this.type = set(clusterK, value)

  /**
    * Param for imbalance ratio threshold.
    * @group param
    */
  final val densityExponent: Param[Int] = new Param[Int](this, "densityExponent", "density exponent, default is the number of features if set to 0")

  /** @group getParam */
  final def setDensityExponent(value: Int): this.type = set(densityExponent, value)

  /**
    * Param for imbalance ratio threshold.
    * @group param
    */
  final val imbalanceRatioThreshold: Param[Double] = new Param[Double](this, "imbalanceRatioThreshold", "imbalance ratio threshold")

  /** @group getParam */
  final def setImbalanceRatioThreshold(value: Double): this.type = set(imbalanceRatioThreshold, value)

  /**
    * Param for knn brute force size
    * @group param
    */
  final val bruteForceSize: Param[Int] = new Param[Int](this, "bruteForceSize", "knn brute force size")

  /** @group getParam */
  final def setBruteForceSize(value: Int): this.type = set(bruteForceSize, value)

  setDefault(clusterK -> 5, imbalanceRatioThreshold -> 10.0, bruteForceSize -> 100, densityExponent -> 0)
}

/** Transformer */
class KMeansSMOTEModel private[ml](override val uid: String) extends Model[KMeansSMOTEModel] with KMeansSMOTEModelParams {
  def this() = this(Identifiable.randomUID("kmeanssmote"))

  private def getSingleDistance(x: Array[Double], y: Array[Double]): Double = {
    var distance = 0.0
    for(index<-x.indices) {
      distance += (x(index) -  y(index)) *(x(index) - y(index))
    }
    distance
  }

  private def getImbalancedRatio(spark: SparkSession, data: Dataset[Row], minClassLabel: Double): Double = {
    val minorityCount = data.filter(data($(labelCol)) === minClassLabel).count
    val majorityCount = data.filter(data($(labelCol)) =!= minClassLabel).count
    (majorityCount + 1) / (minorityCount + 1).toDouble
  }

  private def getInstanceAverageDistance(vector: Array[Double], vectors: Array[Array[Double]]): Double ={
    vectors.map(x=>getSingleDistance(vector, x)).sum / (vectors.length - 1)
  }

  private def getAverageDistance(df: DataFrame): Double ={
    val collected = df.select($(featuresCol)).collect.map(x=>x(0).asInstanceOf[DenseVector].toArray)
    collected.map(x=>getInstanceAverageDistance(x, collected)).sum / collected.length.toDouble
  }

  def sampleCluster(df: DataFrame, samplesToAdd: Int): DataFrame ={
    if(df.count < $(bruteForceSize)) {
      val r = new SMOTE
      val model = r.fit(df).setBalanceThreshold(0.0).setTopTreeSize(df.count.toInt).setTopTreeLeafSize(df.count.toInt)
        .setSubTreeLeafSize($(bruteForceSize)).setK($(k))
      model.oversample(df, samplesToAdd)
    } else {
      val r = new SMOTE
      val model = r.fit(df).setBalanceThreshold($(balanceThreshold)).setTopTreeSize($(topTreeSize))
        .setTopTreeLeafSize($(topTreeLeafSize)).setSubTreeLeafSize($(subTreeLeafSize)).setK($(k))
      model.oversample(df, samplesToAdd)
    }
  }

  // fixme - issue with some parameter settings resulting in no clusters for oversampling
  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val numberOfFeatures = dataset.select($(featuresCol)).take(1)(0)(0).asInstanceOf[DenseVector].size
    val spark = dataset.sparkSession

    // STEP 1
    val kmeans = new KMeans().setK($(clusterK))
    val model = kmeans.fit(dataset)
    val predictions = model.transform(dataset)

    val clusters = (0 until $(clusterK)).map(x=>predictions.filter(predictions("prediction")===x).drop("prediction")).toArray

    val imbalancedRatios: Array[Double] = clusters.map(x=>getImbalancedRatio(spark, x, minorityClassLabel))

    val filteredClusters = (0 until $(clusterK)).map(x=>(imbalancedRatios(x), clusters(x)))
      .filter(x=>x._1 < $(imbalanceRatioThreshold)).map(x=>x._2).map(x=>x.filter(x($(labelCol))===minorityClassLabel))
      .filter(x=>x.count() > 0)

    val averageDistances = filteredClusters.indices.map(x=>getAverageDistance(filteredClusters(x))).toArray

    val densityExp = if($(densityExponent) > 0) {
      $(densityExponent)
    } else {
      numberOfFeatures
    }
    val densities = filteredClusters.indices.map(x=>filteredClusters(x).count.toDouble / Math.pow(averageDistances(x), densityExp))
    val sparsities = densities.indices.map(x=>1/densities(x))
    val clusterWeights = sparsities.indices.map(x=>sparsities(x)/sparsities.sum)
    val clusterSamples = clusterWeights.indices.map(x=>(samplesToAdd*clusterWeights(x)).toInt)

    val clusterExamples = filteredClusters.indices.map(x=>sampleCluster(filteredClusters(x), clusterSamples(x))).toArray

    clusterExamples.reduce(_ union _)
  }

    override def transform(dataset: Dataset[_]): DataFrame = {
    //FIXME - add warning about missing clusters

      val indexer = new StringIndexer()
        .setInputCol($(labelCol))
        .setOutputCol("labelIndexed")

      val datasetIndexed = indexer.fit(dataset).transform(dataset)
        .withColumnRenamed($(labelCol), "originalLabel")
        .withColumnRenamed("labelIndexed",  $(labelCol))

      val labelMap = datasetIndexed.select("originalLabel",  $(labelCol)).distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
      val labelMapReversed = labelMap.map(x=>(x._2, x._1))

      val datasetSelected = datasetIndexed.select($(labelCol), $(featuresCol))
      val counts = getCountsByClass(datasetSelected.sparkSession, $(labelCol), datasetSelected.toDF).sort("_2")
      val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
      val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

      val samplingMapConverted: Map[Double, Double] = getSamplingMap($(samplingRatios), labelMap)
      val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)
      val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected.filter(datasetSelected($(labelCol))===clsList(x))))
        .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, x._2.count, majorityClassCount, samplingMapConverted)))

      val balancedDF = if($(oversamplesOnly)) {
        clsDFs.reduce(_ union _)
      } else {
        datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
      }
      val restoreLabel = udf((label: Double) => labelMapReversed(label))

      balancedDF.withColumn("originalLabel", restoreLabel(balancedDF.col($(labelCol)))).drop($(labelCol))
        .withColumnRenamed("originalLabel",  $(labelCol))
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
  def this() = this(Identifiable.randomUID("kmeanssmote"))

  override def fit(dataset: Dataset[_]): KMeansSMOTEModel = {
    val model = new KMeansSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): KMeansSMOTE = defaultCopy(extra)

}

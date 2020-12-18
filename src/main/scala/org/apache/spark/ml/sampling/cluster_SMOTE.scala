package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, getSamplesToAdd, calculateToTreeSize}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random


/** Transformer Parameters*/
private[ml] trait ClusterSMOTEModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {
  /**
    * Param for kNN k-value.
    * @group param
    */
  final val clusterK: Param[Int] = new Param[Int](this, "clusterK", "cluster k-value for kNN")

  /** @group getParam */
  final def setClusterK(value: Int): this.type = set(clusterK, value)

  setDefault(clusterK -> 5)
}

/** Transformer */
class ClusterSMOTEModel private[ml](override val uid: String) extends Model[ClusterSMOTEModel] with ClusterSMOTEModelParams {
  def this() = this(Identifiable.randomUID("clusterSmote"))

  var knnClusters: Array[Array[Row]] = Array[Array[Row]]()
  var knnClusterCounts: Array[Int] = Array[Int]()

  def createSample(clusterId: Int): DenseVector ={
    val row = knnClusters(clusterId)(Random.nextInt(knnClusterCounts(clusterId)))
    val features = row(1).asInstanceOf[mutable.WrappedArray[DenseVector]]
    val aSample = features(0).toArray
    val bSample = features(Random.nextInt(Math.min(features.length, $(k) + 1))).toArray
    val offset = Random.nextDouble()

    Vectors.dense(Array(aSample, bSample).transpose.map(x=>x(0) + offset * (x(1)-x(0)))).toDense
  }

  def calculateKnnByCluster(spark: SparkSession, df: DataFrame): DataFrame ={
    import spark.implicits._

    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), df.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    // FIXME - pick solution for buffer issues
    if(model.getBufferSize < 0.0) {
      val model = new KNN().setFeaturesCol($(featuresCol))
        .setTopTreeSize(calculateToTreeSize($(topTreeSize), df.count()))
        .setTopTreeLeafSize($(topTreeLeafSize))
        .setSubTreeLeafSize($(subTreeLeafSize))
        .setK($(k) + 1) // include self example
        .setAuxCols(Array($(labelCol), $(featuresCol)))
        .setBalanceThreshold($(balanceThreshold))
      val f = model.fit(df)
      f.transform(df).withColumn("neighborFeatures", $"neighbors.features")
    } else {
      val f = model.fit(df)
      f.transform(df).withColumn("neighborFeatures", $"neighbors.features")
    }
  }

  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame ={
    val spark = dataset.sparkSession

    val minorityDF = dataset.filter(dataset($(labelCol))===minorityClassLabel)
    val kMeans = new KMeans().setK($(clusterK))
    val model = kMeans.fit(minorityDF)
    val predictions = model.transform(minorityDF)

    val clusters = (0 until $(clusterK)).map(x=>predictions.filter(predictions("prediction")===x)).filter(x=>x.count()>0).toArray

    // knn for each cluster
    knnClusters = clusters.map(x=>calculateKnnByCluster(spark, x).select($(labelCol), "neighborFeatures").collect).filter(x=>x.length > 0)
    knnClusterCounts = knnClusters.map(x=>x.length)

    val randomIndicies = (0 until samplesToAdd).map(_ => Random.nextInt(clusters.length))
    val addedSamples = randomIndicies.map(x=>Row(minorityClassLabel, createSample(x))).toArray

    spark.createDataFrame(dataset.sparkSession.sparkContext.parallelize(addedSamples), dataset.schema)
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

  override def copy(extra: ParamMap): ClusterSMOTEModel = {
    val copied = new ClusterSMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}

/** Estimator Parameters*/
private[ml] trait ClusterSMOTEParams extends ClusterSMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class ClusterSMOTE(override val uid: String) extends Estimator[ClusterSMOTEModel] with ClusterSMOTEParams {
  def this() = this(Identifiable.randomUID("clusterSmote"))

  override def fit(dataset: Dataset[_]): ClusterSMOTEModel = {
    val model = new ClusterSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ClusterSMOTE = defaultCopy(extra)

}

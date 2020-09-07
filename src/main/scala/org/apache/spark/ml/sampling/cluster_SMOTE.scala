package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.knn.KNN
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
private[ml] trait ClusterSMOTEModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class ClusterSMOTEModel private[ml](override val uid: String) extends Model[ClusterSMOTEModel] with ClusterSMOTEModelParams {
  def this() = this(Identifiable.randomUID("clusterSmote"))

  val knnK = 5
  var knnClusters: Array[Array[Row]] = Array[Array[Row]]()
  var knnClusterCounts: Array[Int] = Array[Int]()

  def createSample(clusterId: Int): DenseVector ={
    val row = knnClusters(clusterId)(Random.nextInt(knnClusterCounts(clusterId)))
    val features = row(1).asInstanceOf[mutable.WrappedArray[DenseVector]]

    val aSample = features(0).toArray
    val bSample = features(Random.nextInt(knnK + 1)).toArray
    val offset = Random.nextDouble()

    /// FIXME - check ALL distance calculations
    Vectors.dense(Array(aSample, bSample).transpose.map(x=>x(0) + offset * (x(1)-x(0)))).toDense
  }

  def calculateKnnByCluster(spark: SparkSession, df: DataFrame): DataFrame ={
    df.show()
    import spark.implicits._

    val leafSize = 10 // FIXME
    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(2) /// FIXME - check? // df.count().toInt / 2
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(knnK + 1) // include self example
      .setAuxCols(Array("label", "features"))
    println(model.getBalanceThreshold)
    println(model.getBufferSize)

    if(model.getBufferSize < 0.0) {
      val model = new KNN().setFeaturesCol("features")
        .setTopTreeSize(2) /// FIXME - check? df.count().toInt / 8
        .setTopTreeLeafSize(leafSize)
        .setSubTreeLeafSize(leafSize)
        .setBalanceThreshold(0.0) // Fixes issue with smaller clusters
        .setK(knnK + 1) // include self example
        .setAuxCols(Array("label", "features"))
      val f = model.fit(df)
      f.transform(df).withColumn("neighborFeatures", $"neighbors.features")
    } else {
      val f = model.fit(df)
      f.transform(df).withColumn("neighborFeatures", $"neighbors.features")
    }
  }

  def oversampleClass(dataset: Dataset[_], minorityClassLabel: String, samplesToAdd: Int): DataFrame ={
    val clusterK = 5 // FIXME
    println("Samples to add: " + samplesToAdd)
    val spark = dataset.sparkSession

    val minorityDF = dataset.filter(dataset("label")===minorityClassLabel)

    val kValue = clusterK // Math.min(minorityClassLabel.toInt, clusterK)
    val kmeans = new KMeans().setK(kValue).setSeed(1L) // FIXME - fix seed
    val model = kmeans.fit(minorityDF)
    val predictions = model.transform(minorityDF)

    val clusters = (0 until clusterK).map(x=>predictions.filter(predictions("prediction")===x)).filter(x=>x.count()>0).toArray

    // knn for each cluster
    knnClusters =  clusters.map(x=>calculateKnnByCluster(spark, x).select("label", "neighborFeatures").collect)
    knnClusterCounts = knnClusters.map(x=>x.length)

    val randomIndicies = (0 until samplesToAdd).map(_ => Random.nextInt(clusters.length))
    val addedSamples = randomIndicies.map(x=>(0.toLong, minorityClassLabel.toInt, createSample(x))).toArray

    val dfAddedSamples = spark.createDataFrame(spark.sparkContext.parallelize(addedSamples))
      .withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    dataset.printSchema()
    dfAddedSamples.printSchema()
    dfAddedSamples
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()

    val counts = getCountsByClass(df.sparkSession, "label", df).sort("_2")
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val minorityClasses = counts.collect.map(x=>(x(0).toString, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)
    val results: DataFrame = minorityClasses.map(x=>oversampleClass(dataset, x._1, majorityClassCount - x._2)).reduce(_ union _).union(dataset.toDF())

    println("dataset: " + dataset.count)
    println("added: " + results.count)

    results
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
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): ClusterSMOTEModel = {
    val model = new ClusterSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ClusterSMOTE = defaultCopy(extra)

}

package org.apache.spark.ml.sampling

import org.apache.spark.ml._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StringIndexer
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
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, getSamplesToAdd}
import org.apache.spark.ml.sampling.utils.getCountsByClass

import scala.collection.mutable
import org.apache.spark.mllib.linalg.MatrixImplicits._
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, DenseVector => OldDenseVector, Matrices => OldMatrices, Vector => OldVector, Vectors => OldVectors}

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
  final val imbalanceRatioThreshold: Param[Double] = new Param[Double](this, "imbalanceRatioThreshold", "imbalance ratio threshold")

  /** @group getParam */
  final def setImbalanceRatioThreshold(value: Double): this.type = set(imbalanceRatioThreshold, value)

  setDefault(clusterK -> 5, imbalanceRatioThreshold -> 10.0)
}

/** Transformer */
class KMeansSMOTEModel private[ml](override val uid: String) extends Model[KMeansSMOTEModel] with KMeansSMOTEModelParams {
  def this() = this(Identifiable.randomUID("kmeanssmote"))

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
  /* private def sampleCluster(df: DataFrame, cls: Int, samplesToAdd: Int): Array[Row] = {

    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize($(topTreeSize))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel: KNNModel = model.fit(df)
    val neighborsDF = fitModel.transform(df)
    neighborsDF.show

    val collected = neighborsDF.collect()
    val count = df.count.toInt

    // examples/k neighbors/features of Doubles
    val examples: Array[Array[Array[Double]]] = collected.map(x=>x(4).toString.substring(13, x(4).toString.length - 3).split("]], ").map(x=>x.split(""",\[""")(1)).map(y=>y.split(",").map(z=>z.toDouble))) // FIMXE
    val randomIndicies = (0 to samplesToAdd).map(_ => Random.nextInt(count))

    val syntheticExamples = randomIndicies.par.map(x=>getSmoteExample(examples(x))).toArray

    syntheticExamples.map(x=>Row(0, cls, Vectors.dense(x)))
  } */

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
    // FIXME - use dataset schema access method
    val collected: Array[Array[Double]] =  data.withColumn($(featuresCol), toArrUdf(col($(featuresCol)))).select($(featuresCol)).collect().map(x=>x.toString.substring(14, x.toString.length-2).split(",").map(x=>x.toDouble)) // FIXME
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
    if(df.count > 1) {
      val r = new SMOTE
      val model = r.fit(df).setBalanceThreshold(0.0) // FIXME - pass parameters
      model.oversample(df, samplesToAdd)
    } else {
      df
    }
  }

  // fixme - issue with some parameter settings resulting in no clusters for oversampling
  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    println("Adding samples... " + samplesToAdd)

    val numberOfFeatures = dataset.select($(featuresCol)).take(1)(0)(0).asInstanceOf[DenseVector].size
    println("number of Features : " + numberOfFeatures)

    val spark = dataset.sparkSession

    // STEP 1
    val kmeans = new KMeans().setK($(clusterK))//.setSeed(1L) // FIXME - fix seed
    val model = kmeans.fit(dataset)
    val predictions = model.transform(dataset)

    val clusters = (0 until $(clusterK)).map(x=>predictions.filter(predictions("prediction")===x).drop("prediction")).toArray

    val imbalancedRatios: Array[Double] = clusters.map(x=>getImbalancedRatio(spark, x, minorityClassLabel))

    val filteredClusters = (0 until $(clusterK)).map(x=>(imbalancedRatios(x), clusters(x))).filter(x=>x._1 < $(imbalanceRatioThreshold)).map(x=>x._2).map(x=>x.filter(x($(labelCol))===minorityClassLabel))
    println("^^^^^^ legnth: " + filteredClusters.length)

    val averageDistances = filteredClusters.indices.map(x=>getAverageDistance(filteredClusters(x))).toArray
    for(x<-averageDistances) {
      println("avg distance: " + x)
    }

    for(x<-filteredClusters.indices) {
      println("cluster count: " + filteredClusters(x).count)
    }
    // FIXME, this doesnt really work with high dimensional data
    val densities = filteredClusters.indices.map(x=>filteredClusters(x).count.toDouble / Math.pow(averageDistances(x), numberOfFeatures))
    for(x<-densities) {
      println("densities: " + x)
    }
    val sparsities = densities.indices.map(x=>1/densities(x))

    for(x<-sparsities) {
      println("sparsities: " + x)
    }

    val clusterWeights = sparsities.indices.map(x=>sparsities(x)/sparsities.sum)

    for(x<-clusterWeights) {
      println("cluster weight: " + x)
    }

    val clusterSamples = clusterWeights.indices.map(x=>(samplesToAdd*clusterWeights(x)).toInt)

    println("Counts")
    for(x<-clusterSamples) {
      println(x)
    }

    //df.union(filteredClusters.indices.map(x=>sampleCluster(filteredClusters(x), clusterSamples(x))).reduce(_ union _))
    val xx = filteredClusters.indices.map(x=>sampleCluster(filteredClusters(x), clusterSamples(x))).toArray // .reduce(_ union _)
    println("length: " + xx.length)
    for(x<-xx) {
      println("count: " + x.count())
    }

    xx.reduce(_ union _)
  }

    override def transform(dataset: Dataset[_]): DataFrame = {
    // FIXME - add de -density parameter, currently auto calculated
    //FIXME - add warning about missing clusters

    // val densityExponent = 10 // FIXME - number of features, make parameter as well?

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

      val balancedDF = datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
      val restoreLabel = udf((label: Double) => labelMapReversed(label))

      balancedDF.withColumn("originalLabel", restoreLabel(balancedDF.col($(labelCol)))).drop($(labelCol))
        .withColumnRenamed("originalLabel",  $(labelCol)).repartition(1)
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

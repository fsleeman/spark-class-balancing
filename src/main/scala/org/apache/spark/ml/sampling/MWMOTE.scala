package org.apache.spark.ml.sampling

import org.apache.commons.math3.random.{RandomDataGenerator, UniformRandomGenerator}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, calculateToTreeSize}
import org.apache.spark.ml.sampling.utils.{getCountsByClass, getMatchingClassCount}
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.mllib.random.RandomRDDs.randomVectorRDD
import org.apache.spark.mllib.random.UniformGenerator


import scala.collection.mutable
import scala.util.Random
import org.apache.spark.sql.functions._
import org.apache.spark.ml.sampling.utils.pointDifference
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.StructType


/** Transformer Parameters*/
private[ml] trait MWMOTEModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {
  /**
    * Param for kNN k1-value.
    * @group param
    */
  final val k1: Param[Int] = new Param[Int](this, "k1", "k1-value for kNN")

  /** @group getParam */
  final def setK1(value: Int): this.type = set(k1, value)

  /**
    * Param for kNN k2-value.
    * @group param
    */
  final val k2: Param[Int] = new Param[Int](this, "k2", "k2-value for kNN")

  /** @group getParam */
  final def setK2(value: Int): this.type = set(k2, value)

  /**
    * Param for kNN k1-value.
    * @group param
    */
  final val k3: Param[Int] = new Param[Int](this, "k3", "k3-value for kNN")

  /** @group getParam */
  final def setK3(value: Int): this.type = set(k3, value)

  /**
    * Param for kNN k-value.
    * @group param
    */
  final val clusterK: Param[Int] = new Param[Int](this, "clusterK", "cluster k-value for kNN")

  /** @group getParam */
  final def setClusterK(value: Int): this.type = set(clusterK, value)

  /**
    * Param for kNN k-value.
    * @group param
    */
  final val cMax: Param[Double] = new Param[Double](this, "cMax", "cMax value")

  /** @group getParam */
  final def setCMax(value: Double): this.type = set(cMax, value)

  /**
    * Param for kNN k-value.
    * @group param
    */
  final val cfth: Param[Double] = new Param[Double](this, "cfth", "Cf_th value")

  /** @group getParam */
  final def setCfth(value: Double): this.type = set(cfth, value)

  setDefault(k1->$(k), k2->$(k), k3->$(k), clusterK->10, cMax->3.0, cfth->50.0)
}

/** Transformer */
class MWMOTEModel private[ml](override val uid: String) extends Model[MWMOTEModel] with MWMOTEModelParams {
  def this() = this(Identifiable.randomUID("mwmote"))

  def explodeNeighbors(labels: mutable.WrappedArray[Double], features: mutable.WrappedArray[DenseVector]): Array[(Double, DenseVector)] = {
    val len = labels.length
    (1 until len).map(x=>(labels(x), features(x))).toArray
  }

  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val df = dataset.toDF
    import df.sparkSession.implicits._
    val spark = df.sparkSession

    val featureLength = df.select("features").take(1)(0)(0).asInstanceOf[DenseVector].size

    val Smin = df.filter(df($(labelCol)) === minorityClassLabel)
    val Smaj = df.filter(df($(labelCol)) =!= minorityClassLabel)

    /*1) find k1 NNs of minority examples
      2) from minority examples, remove examples with no minority example neighbors
      3) find k2 nearest majority examples for each in minority data
      4) union the nearest neighbor majority examples to build majority borderline set
      5) for each borderline majority example, find the k3 nearest neighbors from the minority examples
      6) union the nearest neighbors minority examples to build minority borderline set
      7) for majority/minority sets, compute the information weight
      8) for borderline minority, computer the selection weight
      9)  convert each to selection probability
      10) find clusters of minority examples
      11) initialize set
      12) for 1 to N
        a) selection x from borderline minority, by probability distribution
        b) selection sample y from cluster
        c) generate one synthetic example
        d) add example
    */


    /** 1 **/
    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), df.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k1) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val f1: KNNModel = model.fit(df)
    val SminNN = f1.transform(Smin)

    /** 2 **/
    val Sminf = SminNN.withColumn("nnMinorityCount", getMatchingClassCount($"neighbors.label", col($(labelCol)))).filter("nnMinorityCount > 0")


    /** 3 **/
    val model2 = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), Smaj.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k2))
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val f2: KNNModel = model2.fit(Smaj)
    val Nmaj = f2.transform(Sminf.drop("neighbors", "nnMinorityCount"))

    /** 4 **/
    val explodedNeighbors = Nmaj.select("neighbors").withColumn($(labelCol), $"neighbors.label")
      .withColumn($(featuresCol), $"neighbors.features")
      .collect.flatMap(x=>explodeNeighbors(x(1).asInstanceOf[mutable.WrappedArray[Double]], x(2).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val Sbmaj = spark.sparkContext.parallelize(explodedNeighbors).toDF($(labelCol), $(featuresCol)).distinct()


    /** 5 **/
    val model3 = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), Smin.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k3))
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val f3: KNNModel = model3.fit(Smin)
    val Nmin = f3.transform(Sbmaj.drop("neighbors", "nnMinorityCount"))


    /** 6 **/
    val explodedNeighbors2 = Nmin.select("neighbors").withColumn($(labelCol), $"neighbors.label")
      .withColumn($(featuresCol), $"neighbors.features")
      .collect.flatMap(x=>explodeNeighbors(x(1).asInstanceOf[mutable.WrappedArray[Double]], x(2).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    // FIXME - correct Simin creation
    val Simin = spark.sparkContext.parallelize(explodedNeighbors2).toDF($(labelCol), $(featuresCol)).distinct()


    /** 7 **/
    val SbmajCollected: Array[DenseVector] = Sbmaj.select($(featuresCol)).collect().map(x=>x(0).asInstanceOf[DenseVector])

    val getClosenessFactors = udf((minorityExample: DenseVector) => {
      def calculateClosenessFactor(x: DenseVector, y: DenseVector, l: Int): Double ={
        val distance = pointDifference(y.toArray, x.toArray) / l.toDouble
        // FIXME- should this be on the order of the number of features?

        val numerator = if(1/distance <= $(cfth)) {
          1/distance
        } else {
          $(cfth)
        }
        (numerator / $(cfth)) * $(cMax)
      }
      SbmajCollected.map(SbmajValue=>calculateClosenessFactor(minorityExample, SbmajValue, featureLength))
    })

    val closenessFactors = Simin.withColumn("closeness", getClosenessFactors(Simin($(featuresCol)))) // result is majority data with minority closeness values
    val calculateInformationWeights = udf((closenessFactors: mutable.WrappedArray[Double]) => {
      closenessFactors.toArray.map(x=>x*x/closenessFactors.toArray.sum)
    })

    val informationWeights = closenessFactors.withColumn("informationWeights", calculateInformationWeights(closenessFactors("closeness")))


    /* 8 */
    val calculateSelectionWeights = udf((informationWeights: mutable.WrappedArray[Double]) => {
      informationWeights.toArray.sum
    })
    val selectionWeights = informationWeights.withColumn("selectionWeights", calculateSelectionWeights(informationWeights("informationWeights")))


    /* 9 */
    val selectionWeightSum = selectionWeights.select("selectionWeights").collect().map(x=>x(0).asInstanceOf[Double]).sum

    val calculateSelectionProbabilities = udf((selectionWeight: Double) => {
      selectionWeight / selectionWeightSum
    })
    val selectionProbabilities = selectionWeights.withColumn("selectionProbabilities", calculateSelectionProbabilities(selectionWeights("selectionWeights")))


    /* 10 */
    val kmeans = new KMeans().setK($(clusterK))
    val kMeansModel = kmeans.fit(Smin)
    val SiminKMeansClusters = kMeansModel.transform(selectionProbabilities)

    /* 11 */
    // Somin = Smin -- don't do it here


    /* 12 */

    val SiminClustersCollected = SiminKMeansClusters.select($(labelCol), $(featuresCol), "selectionProbabilities", "prediction")

    val partitionWindow = Window.partitionBy($"label").orderBy($"selectionProbabilities".desc).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    val sumTest = sum($"selectionProbabilities").over(partitionWindow)
    val runningTotals = SiminClustersCollected.select($"*", sumTest as "running_total")
    val runningTotalsCollected = runningTotals.collect()
    val groupedClusters = (0 until $(clusterK)).map(x=>runningTotalsCollected.filter(_(3)==x))

    def generateExample(randomNumbers: Array[Double]): (Double, DenseVector) = {

      val randomIndexPoint = randomNumbers(0)
      val selectedRow = runningTotalsCollected.filter(x=>x(4).asInstanceOf[Double]>=randomIndexPoint)(0)

      val label = selectedRow(0).asInstanceOf[Double]
      val features = selectedRow(1).asInstanceOf[DenseVector]
      val cluster = selectedRow(3).asInstanceOf[Int]

      val randomIndex = (groupedClusters(cluster).length * randomNumbers(1)).toInt
      val randomExample = groupedClusters(cluster)(randomIndex)(1).asInstanceOf[DenseVector]

      val alpha = randomNumbers(2)
      val synthetic = Array(features.toArray, randomExample.toArray).transpose.map(x=> x(0) + alpha * (x(1) - x(0)))

      (label, Vectors.dense(synthetic).toDense)
    }

    val generator = new UniformGenerator()

    val syntheticDF = if(runningTotalsCollected.length > 0) {
      val randomNumbers = randomVectorRDD(spark.sparkContext, generator, samplesToAdd, 3, dataset.rdd.getNumPartitions)
      val syntheticExamples = randomNumbers.map(x=>generateExample(x.toArray))

      spark.createDataFrame(syntheticExamples).toDF()
        .withColumnRenamed("_1",$(labelCol))
        .withColumnRenamed("_2",$(featuresCol))
    } else {
      val r = new RandomOversample()
      val model = r.fit(Smin).setSingleClassOversampling(samplesToAdd)
      model.transform(Smin).toDF()
    }

    syntheticDF
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
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString.toDouble
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val minorityClasses = counts.collect.map(x=>(x(0).toString.toDouble, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)
    val balancedDF: DataFrame = minorityClasses.map(x=>oversample(datasetSelected, x._1, majorityClassCount - x._2)).reduce(_ union _).union(datasetSelected.toDF()).select($(labelCol), $(featuresCol))

    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balancedDF.withColumn("originalLabel", restoreLabel(balancedDF.col($(labelCol)))).drop($(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))//.repartition(1)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): MWMOTEModel = {
    val copied = new MWMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait MWMOTEParams extends MWMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class MWMOTE(override val uid: String) extends Estimator[MWMOTEModel] with MWMOTEParams {
  def this() = this(Identifiable.randomUID("mwmote"))

  override def fit(dataset: Dataset[_]): MWMOTEModel = {
    val model = new MWMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): MWMOTE = defaultCopy(extra)

}

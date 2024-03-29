package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.Utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, calculateToTreeSize, getSamplesToAdd, getSamplingMap}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, lit, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable
import scala.util.Random


/** Transformer Parameters*/
private[ml] trait ANSModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {

  /**
    * Param for kNN top tree leaf size.
    * @group param
    */
  final val cMaxRatio: Param[Double] = new Param[Double](this, "cMaxRatio", "C_Max ratio (0, 1]")

  /** @group getParam */
  final def setCMaxRatio(value: Double): this.type = set(cMaxRatio, value)

  /**
    * Param for kNN top tree leaf size.
    * @group param
    */
  final val distanceNeighborLimit: Param[Int] = new Param[Int](this, "distanceNeighborLimit", "limit of how many distance based neighbors to keep, default=100, using 0 will returns all")

  /** @group getParam */
  final def setdDstanceNeighborLimit(value: Int): this.type = set(distanceNeighborLimit, value)

  setDefault(cMaxRatio -> 0.25, distanceNeighborLimit -> 100)
}

/** Transformer */
class ANSModel private[ml](override val uid: String) extends Model[ANSModel] with ANSModelParams {
  def this() = this(Identifiable.randomUID("ans"))

  def createSample(row: Row): Array[Row] = {
    val label = row(0).toString.toDouble
    val features: Array[Double] = row(1).asInstanceOf[DenseVector].toArray
    val neighbors = row(2).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail
    val samplesToAdd = row(3).toString.toInt

    def addSample(): Row ={
      val randomNeighbor: Array[Double] = neighbors(Random.nextInt(neighbors.length)).toArray
      val gap = Random.nextDouble()
      val syntheticExample = Vectors.dense(Array(features, randomNeighbor).transpose.map(x=>x(0) + gap * (x(1)-x(0)))).toDense
      Row(label, syntheticExample)
    }

    (0 until samplesToAdd).map(_=>addSample()).toArray
  }

  private val getNearestNeighborDistance = udf((distances: mutable.WrappedArray[Double]) => {
    distances(1)
  })


  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val minorityDF = dataset.filter(dataset($(labelCol)) === minorityClassLabel)
    val majorityDF = dataset.filter(dataset($(labelCol)) =!= minorityClassLabel)

    val C_max = Math.ceil($(cMaxRatio) * dataset.count()).toInt

    val minorityKnnModel = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), minorityDF.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK(1 + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val minorityKnnFit: KNNModel = minorityKnnModel.fit(minorityDF).setDistanceCol("distances")
    val neighborDistances = minorityKnnFit.transform(minorityDF).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")

    val firstPosNeighborDistances = neighborDistances.withColumn("closestPosDistance", getNearestNeighborDistance($"distances")).drop("distances", "neighborFeatures")

    val majorityKnnModel: KNN = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), majorityDF.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val majorityKnnFit: KNNModel = majorityKnnModel.fit(majorityDF).setDistanceCol("distances").setMaxDistanceCol("closestPosDistance").setQueryByDistance(true).setDistanceNeighborLimit($(distanceNeighborLimit))
    val majorityNeighbors = majorityKnnFit.transform(firstPosNeighborDistances).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")

    val getRadiusNeighbors = udf((distances: mutable.WrappedArray[Double]) => {
      distances.length - 1 // ignore self neighbor
    })

    val outBorder = majorityNeighbors.withColumn("outBorder", getRadiusNeighbors($"distances"))
        val outBorderArray = outBorder.select("outBorder").collect().map(x => x(0).asInstanceOf[Int])

   var previous_number_of_outcasts = -1
   var C = 1

   import scala.util.control._
    val loop = new Breaks
    loop.breakable {
      for (c <- 1 until C_max) {
        val number_of_outcasts = outBorderArray.filter(x => x >= c).sum
        if (Math.abs(number_of_outcasts - previous_number_of_outcasts) == 0) {
          C = c
          if(outBorder.filter(outBorder("outBorder") < C).count > 0) {
            loop.break()
          }
        }
        previous_number_of_outcasts = number_of_outcasts
      }
    }

    val Pused = outBorder.filter(outBorder("outBorder") < C).drop("distances", "neighborFeatures", "outBorder")
    val maxClosestPosDistance = Pused.select("closestPosDistance").collect().map(x=>x(0).toString.toDouble).max

    val PusedKnnModel: KNN = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), Pused.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val PusedSearchDistance = Pused.withColumn("searchDistance", lit(maxClosestPosDistance))

    val PusedKnnFit: KNNModel = PusedKnnModel.fit(PusedSearchDistance).setDistanceCol("distances").setMaxDistanceCol("searchDistance").setQueryByDistance(true).setDistanceNeighborLimit($(distanceNeighborLimit))
    val PusedDistances = PusedKnnFit.transform(PusedSearchDistance).withColumn("neighborFeatures", $"neighbors.features")
      .drop("neighbors")

    val PusedDistancesNeighborCount = PusedDistances.withColumn("neighborCount", getRadiusNeighbors($"distances"))

    val neighborCountSum = PusedDistancesNeighborCount.select("neighborCount").collect().map(x=>x(0).toString.toInt).sum.toDouble

    val getSamplesToAdd = udf((count: Int) => {
      Math.ceil((count / neighborCountSum) * samplesToAdd).toInt
    })

    val generatedSampleCounts = PusedDistancesNeighborCount.withColumn("samplesToAdd", getSamplesToAdd($"neighborCount"))

    val syntheticExamples: Array[Array[Row]] = generatedSampleCounts.drop("closestPosDistance", "searchDistance", "distances", "neighborCount").filter("neighborCount > 0")
      .collect.map(x=>createSample(x))

    val totalExamples: Array[Row] = syntheticExamples.flatMap(x => x.toSeq).take(samplesToAdd)

    spark.createDataFrame(dataset.sparkSession.sparkContext.parallelize(totalExamples), dataset.schema)
  }
    override def transform(dataset: Dataset[_]): DataFrame = {

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

      val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected, x))
        .map(x=>oversample(x._2, x._1,
          getSamplesToAdd(x._1.toDouble, datasetSelected.filter(datasetSelected($(labelCol))===clsList(x._3)).count(),
            majorityClassCount, samplingMapConverted)))

      val balanecedDF = if($(oversamplesOnly)) {
        clsDFs.reduce(_ union _)
      } else {
        datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
      }

      val restoreLabel = udf((label: Double) => labelMapReversed(label))

      balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
        .withColumnRenamed("originalLabel",  $(labelCol))//.repartition(1)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): ANSModel = {
    val copied = new ANSModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait ANSParams extends ANSModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class ANS(override val uid: String) extends Estimator[ANSModel] with ANSParams {
  def this() = this(Identifiable.randomUID("ans"))

  override def fit(dataset: Dataset[_]): ANSModel = {
    val model = new ANSModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ANS = defaultCopy(extra)

}


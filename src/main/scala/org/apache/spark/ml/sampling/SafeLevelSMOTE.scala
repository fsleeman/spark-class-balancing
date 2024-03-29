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
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable
import scala.util.Random
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.StructType


/** Transformer Parameters*/
private[ml] trait SafeLevelSMOTEModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {
  /**
    * Param for kNN k-value.
    * @group param
    */
  final val samplingCorrectionRate: Param[Double] = new Param[Double](this, "samplingCorrectionRate", "proportion of over/undersampling to allow")

  /** @group getParam */
  final def setSamplingCorrectionRate(value: Double): this.type = set(samplingCorrectionRate, value)

  setDefault(samplingCorrectionRate -> 0.05)
}

/** Transformer */
class SafeLevelSMOTEModel private[ml](override val uid: String) extends Model[SafeLevelSMOTEModel] with SafeLevelSMOTEModelParams {
  def this() = this(Identifiable.randomUID("safeLevel"))

  private val getSafeNeighborCount = udf((array: mutable.WrappedArray[Double], minorityClassLabel: Double) => {
    def isMajorityNeighbor(x1: Double, x2: Double): Int = {
      if(x1 == x2) {
        1
      } else {
        0
      }
    }
    array.tail.map(x=>isMajorityNeighbor(minorityClassLabel, x)).sum
  })

  private val getRatios = udf({(pClassCount: Int, nClassCounts: mutable.WrappedArray[Int]) => Double
    def getRatio(nClassCount: Int): Double = {
      if(nClassCount != 0) {
        pClassCount.toDouble / nClassCount.toDouble
      } else {
        Double.NaN
      }
    }

    nClassCounts.map(x=>getRatio(x))
  })

  private def generateExamples(label: Double, features: DenseVector, neighborFeatures: mutable.WrappedArray[DenseVector], ratios: mutable.WrappedArray[Double], sampleCount: Int) : Array[Row] ={

    def generateExample(index: Int) : Row ={

      // FIXME -- check these
      val gap = if(ratios(index) == Double.NaN) {
        0
      } else if(ratios(index) == 1) {
        Random.nextDouble()
      } else if(ratios(index) > 1) {
        Random.nextDouble() * (1.0 / ratios(index))
      } else {
        (1.0 - ratios(index)) + Random.nextDouble() * ratios(index)
      }
      val syntheticExample = Vectors.dense(Array(features.toArray, neighborFeatures(index).toArray).transpose.map(x=>x(0) + (x(1) - x(0)) * gap)).toDense

      Row(label, syntheticExample)
    }

    (0 until sampleCount).map(_=>generateExample(Random.nextInt(neighborFeatures.length-1)+1)).toArray
  }

  def calculateNeighborRatioDF(dataset: Dataset[_]): DataFrame = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), dataset.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel: KNNModel = model.fit(dataset)
    val fullNearestNeighborDF = fitModel.transform(dataset)

    val dfNeighborRatio = fullNearestNeighborDF.withColumn("pClassCount",
      getSafeNeighborCount($"neighbors.label", $"label")).drop("neighbors")

    val model2 = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), dataset.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol), "pClassCount"))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel2: KNNModel = model2.fit(dfNeighborRatio)
    fitModel2.transform(dfNeighborRatio)
  }

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


  def oversample(minorityNearestNeighbor: Dataset[_], datasetSchema: StructType, minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val spark = minorityNearestNeighbor.sparkSession
    import spark.implicits._

    val minorityDF = minorityNearestNeighbor.filter(minorityNearestNeighbor($(labelCol)) === minorityClassLabel)
    val minorityFilteredDF = minorityDF.filter(minorityNearestNeighbor("pClassCount")=!=0)

    if(minorityFilteredDF.count() > 0) {
      val minorityRatiosDF = minorityFilteredDF.withColumn("ratios", getRatios($"pClassCount", $"neighbors.pClassCount")).withColumn("neighborsFeatures", $"neighbors.features")
      val samplingRate = Math.ceil(samplesToAdd / minorityRatiosDF.count.toDouble).toInt

      val syntheticExamples: Array[Row] = minorityRatiosDF.collect().map(x=>generateExamples(x(0).asInstanceOf[Double],
        x(1).asInstanceOf[DenseVector], x(5).asInstanceOf[mutable.WrappedArray[DenseVector]],
        x(4).asInstanceOf[mutable.WrappedArray[Double]], samplingRate)).reduce(_ union _ )

      val result = removeNegatives(spark.createDataFrame(spark.sparkContext.parallelize(syntheticExamples), datasetSchema))

      if(syntheticExamples.length * (1.0 + $(samplingCorrectionRate)) > samplesToAdd) {
        result.sample(withReplacement=false, samplesToAdd.toDouble/syntheticExamples.length.toDouble)
      } else if(syntheticExamples.length * (1.0 + $(samplingCorrectionRate)) < samplesToAdd){
        result.sample(withReplacement=true, samplesToAdd.toDouble/syntheticExamples.length.toDouble)
      } else {
        result
      }
    } else {
      val r = new RandomOversample()
      val minorityDFSelected = minorityDF.select($(labelCol), $(featuresCol))
      val model = r.fit(minorityDFSelected).setSingleClassOversampling(samplesToAdd)
      model.transform(minorityDFSelected).toDF()
    }
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
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString.toDouble
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val nearestNeighborDF = calculateNeighborRatioDF(datasetSelected)

    val samplingMapConverted: Map[Double, Double] = getSamplingMap($(samplingRatios), labelMap)
    val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)
    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected, x))
      .map(x=>oversample(nearestNeighborDF, datasetSelected.schema, x._1,
        getSamplesToAdd(x._1.toDouble, datasetSelected.filter(datasetSelected($(labelCol))===clsList(x._3)).count(),
          majorityClassCount, samplingMapConverted)))

    val balancedDF = if($(oversamplesOnly)) {
      clsDFs.reduce(_ union _)
    } else {
      datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    }

    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balancedDF.withColumn("originalLabel", restoreLabel(balancedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): SafeLevelSMOTEModel = {
    val copied = new SafeLevelSMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait SafeLevelSMOTEParams extends SafeLevelSMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class SafeLevelSMOTE(override val uid: String) extends Estimator[SafeLevelSMOTEModel] with SafeLevelSMOTEParams {
  def this() = this(Identifiable.randomUID("safeLevel"))

  override def fit(dataset: Dataset[_]): SafeLevelSMOTEModel = {
    val model = new SafeLevelSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): SafeLevelSMOTE = defaultCopy(extra)

}

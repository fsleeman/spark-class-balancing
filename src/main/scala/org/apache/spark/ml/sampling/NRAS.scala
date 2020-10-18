package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.{Element, getCountsByClass}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, lit, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.knn.{KNN, KNNModel}
import utils.getMatchingClassCount
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, getSamplesToAdd}

import scala.collection.mutable
import scala.util.Random



/** Transformer Parameters*/
private[ml] trait NRASModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {

}

/** Transformer */
class NRASModel private[ml](override val uid: String) extends Model[NRASModel] with NRASModelParams {
  def this() = this(Identifiable.randomUID("nras"))

  // FIXME - reuse from basic SMOTE?
  def getSmoteSample(row: Row): Row = {
    val label = row(0).toString.toDouble
    val features: Array[Double] = row(1).asInstanceOf[DenseVector].toArray
    val neighbors = row(2).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail
    val randomNeighbor: Array[Double] = neighbors(Random.nextInt(neighbors.length)).toArray

    val gap = randomNeighbor.indices.map(_=>Random.nextDouble()).toArray // FIXME - should this be one value instead?
    val syntheticExample = Vectors.dense(Array(features, randomNeighbor, gap).transpose.map(x=>x(0) + x(2) * (x(1)-x(0)))).toDense

    Row(label, syntheticExample)
  }

  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    println("**** samples to add: " + samplesToAdd)
    val spark = dataset.sparkSession
    import spark.implicits._
    println(dataset.take(1)(0))

    dataset.show()
    dataset.printSchema()
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setTol(1e-6)
      .setRegParam(0.01)

    val lrModel = lr.fit(dataset).setProbabilityCol("probability")
    var labelIndex = -1

    for(x <- lrModel.summary.labels.indices) {
      if(lrModel.summary.labels(x) == minorityClassLabel) {
        labelIndex = x
      }
    }

    println("model label index: " + labelIndex + " cls: " + lrModel.summary.labels(labelIndex))

    val propensity = lrModel.transform(dataset)
    propensity.show()
    propensity.printSchema()

    val minorityClassProbablity = udf((probs: DenseVector, i: Int) => probs(i))
    val probs = propensity.withColumn("minorityClassProbability", minorityClassProbablity(propensity("probability"), lit(labelIndex)))
    probs.show

    val addPropensityValue = udf((features: DenseVector, probability: Double) => {
      Vectors.dense(features.toArray :+ probability).toDense
    })

    val updated = probs.withColumn("updatedFeatures", addPropensityValue(probs($(featuresCol)), probs("minorityClassProbability")))
      .drop($(featuresCol)).withColumnRenamed("updatedFeatures", $(featuresCol))
    updated.show

    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize($(topTreeSize))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val f: KNNModel = model.fit(propensity).setDistanceCol("distances")

    val minorityDF = propensity.filter(propensity($(labelCol)) === minorityClassLabel)
    val minorityKnn = f.transform(minorityDF)
    minorityKnn.show

    val threshold = 3 // FIXME - make parameter
    val neighborCounts = minorityKnn.withColumn("sameClassNeighbors", getMatchingClassCount(minorityKnn("neighbors.label"), minorityKnn($(labelCol))))
    val neighborClassFiltered = neighborCounts.filter(neighborCounts("sameClassNeighbors") >= threshold)

    neighborClassFiltered.show

    // Check is filtered data is empty
    val dataForSmote1 = if(neighborClassFiltered.count() > 0) {
      neighborClassFiltered
    } else {
      neighborCounts
    } //withColumn("neighborFeatures", $"neighbors.features").select($(labelCol), $(featuresCol), "neighborFeatures")

    val dataForSmote = dataForSmote1.withColumn("neighborFeatures", $"neighbors.features").select($(labelCol), $(featuresCol), "neighborFeatures")
    println("~~ at dataForSmote")
    dataForSmote.show
    dataForSmote.printSchema()
    val dataForSmoteCollected = dataForSmote.collect()//dataForSmote.withColumn("neighborFeatures", $"neighbors.features").drop("neighbors").collect
    //dataForSmote.withColumn("neighborFeatures", $"neighbors.features").show()
    //dataForSmote.withColumn("neighborFeatures", $"neighbors.features").printSchema()

    println("dataForSmoteCollected: " + dataForSmoteCollected.length)
    val randomIndicies = (0 until samplesToAdd).map(_=>Random.nextInt(dataForSmoteCollected.length))
    val createdSamples = spark.createDataFrame(spark.sparkContext.parallelize(randomIndicies.map(x=>getSmoteSample(dataForSmoteCollected(x)))), dataset.schema)

    println("createSamples "  + createdSamples.count)

    createdSamples
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

    val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)

    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected, x))
      .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, datasetSelected.filter(datasetSelected($(labelCol))===clsList(x._3)).count(), majorityClassCount, $(samplingRatios))))

    val balanecedDF = datasetIndexed.select($(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol)).repartition(1)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): NRASModel = {
    val copied = new NRASModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}

/** Estimator Parameters*/
private[ml] trait NRASParams extends NRASModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class NRAS(override val uid: String) extends Estimator[NRASModel] with NRASParams {
  def this() = this(Identifiable.randomUID("nras"))

  override def fit(dataset: Dataset[_]): NRASModel = {
    val model = new NRASModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): NRAS = defaultCopy(extra)

}

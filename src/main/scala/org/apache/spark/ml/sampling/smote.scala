package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}

import scala.collection.mutable
import scala.util.Random
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.sampling.utilities._
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.{desc, udf}


/** Transformer Parameters*/
private[ml] trait SMOTEModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {

}

/** Transformer */
class SMOTEModel private[ml](override val uid: String) extends Model[SMOTEModel] with SMOTEModelParams {
  def this() = this(Identifiable.randomUID("smote"))

  def getSmoteSample(row: Row): Row = {
    val label = row(0).toString.toDouble
    val features: Array[Double] = row(1).asInstanceOf[DenseVector].toArray
    val neighbors = row(2).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail
    val randomNeighbor: Array[Double] = neighbors(Random.nextInt(neighbors.length)).toArray

    val gap = Random.nextDouble()
    val syntheticExample = Vectors.dense(Array(features, randomNeighbor).transpose.map(x=>x(0) + gap * (x(1) - x(0)))).toDense

    Row(label, syntheticExample)
  }

  def oversample(dataset: Dataset[_], samplesToAdd: Int): DataFrame = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), dataset.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    println("^^^ smote partitions: " + dataset.rdd.getNumPartitions + " dataset size: " + dataset.count())

    val knnModel: KNNModel = model.fit(dataset)
    val nearestNeighborDF = knnModel.transform(dataset)

    val dfCount = nearestNeighborDF.count.toInt
    val randomIndicies = (0 until samplesToAdd).map(_=>Random.nextInt(dfCount))
    val collected = nearestNeighborDF.withColumn("neighborFeatures", $"neighbors.features").drop("neighbors").collect

    spark.createDataFrame(spark.sparkContext.parallelize(randomIndicies.map(x=>getSmoteSample(collected(x)))), dataset.schema)
  }

  override def transform(dataset: Dataset[_]): DataFrame ={

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

    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected.filter(datasetSelected($(labelCol))===clsList(x))))
      .map(x=>oversample(x._2, getSamplesToAdd(x._1.toDouble, x._2.count, majorityClassCount, $(samplingRatios))))

    val balanecedDF = datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): SMOTEModel = {
    val copied = new SMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}

/** Estimator Parameters*/
private[ml] trait SMOTEParams extends SMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class SMOTE(override val uid: String) extends Estimator[SMOTEModel] with SMOTEParams {
  def this() = this(Identifiable.randomUID("smote"))

  override def fit(dataset: Dataset[_]): SMOTEModel = {
    val model = new SMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): SMOTE = defaultCopy(extra)

}

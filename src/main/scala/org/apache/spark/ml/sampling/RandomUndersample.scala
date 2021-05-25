package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.sampling.Utilities.{ClassBalancingRatios, HasLabelCol, getSamplingMap}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.udf


/** Transformer Parameters*/
private[ml] trait RandomUndersampleModelParams extends Params with HasFeaturesCol with HasLabelCol with ClassBalancingRatios {

}

/** Transformer */
class RandomUndersampleModel private[ml](override val uid: String) extends Model[RandomUndersampleModel] with RandomUndersampleModelParams {
  def this() = this(Identifiable.randomUID("randomUndersample"))

  private def getSamplesToKeep(label: Double, sampleCount: Long, minorityClassCount: Int, samplingRatios: Map[Double, Double]): Int ={
    if(samplingRatios contains label) {
      val ratio = samplingRatios(label)
      if(ratio >= 1) {
        sampleCount.toInt
      } else {
        (ratio * sampleCount).toInt
      }
    } else {
      minorityClassCount
    }
  }

  def underSample(df: DataFrame, numSamples: Int): DataFrame = {
    val spark = df.sparkSession

    val underSampleRatio = numSamples / df.count().toDouble
    if (underSampleRatio < 1.0) {
      val currentSamples = df.sample(withReplacement = false, underSampleRatio).collect()
      spark.sqlContext.createDataFrame(spark.sparkContext.parallelize(currentSamples), df.schema)
    }
    else {
      df
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
    val minorityClassCount = counts.orderBy("_2").take(1)(0)(1).toString.toInt

    // FIXME - change sampling method
    val samplingMapConverted: Map[Double, Double] = getSamplingMap($(samplingRatios), labelMap)
    val clsList: Array[Double] = counts.select("_1").collect().map(x=>x(0).toString.toDouble)

    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected.filter(datasetSelected($(labelCol))===clsList(x))))
      .map(x=>underSample(x._2, getSamplesToKeep(x._1.toDouble, x._2.count, minorityClassCount, samplingMapConverted)))

    val balanecedDF = clsDFs.reduce(_ union _)
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))//.repartition(1)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): RandomUndersampleModel = {
    val copied = new RandomUndersampleModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}

/** Estimator Parameters*/
private[ml] trait RandomUndersampleParams extends RandomUndersampleModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class RandomUndersample(override val uid: String) extends Estimator[RandomUndersampleModel] with RandomUndersampleParams {
  def this() = this(Identifiable.randomUID("randomUndersample"))

  override def fit(dataset: Dataset[_]): RandomUndersampleModel = {
    val model = new RandomUndersampleModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): RandomUndersample = defaultCopy(extra)

}




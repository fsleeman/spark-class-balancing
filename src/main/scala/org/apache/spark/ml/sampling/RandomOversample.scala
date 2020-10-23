package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row}

import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.{desc, udf}


/** Transformer Parameters*/
private[ml] trait RandomOversampleModelParams extends Params with HasFeaturesCol with HasLabelCol with ClassBalancingRatios {
  final val singleClassOversamplingSize: Param[Int] = new Param[Int](this, "singleClassOversampling", "samples to add for single class case")

  setDefault(singleClassOversamplingSize, 0)

  /** @group getParam */
  final def setSingleClassOversampling(value: Int): this.type = set(singleClassOversamplingSize, value)
}

/** Transformer */
class RandomOversampleModel private[ml](override val uid: String) extends Model[RandomOversampleModel] with RandomOversampleModelParams {
  def this() = this(Identifiable.randomUID("randomOversample"))

  def oversample(df: DataFrame, numSamples: Int): DataFrame = {
    var samples = Array[Row]() //FIXME - make this more parallel
    val spark = df.sparkSession
    //FIXME - some could be zero if split is too small

    val currentCount = df.count()
    if (0 < currentCount) {
      val currentSamples = df.sample(withReplacement = true, numSamples / df.count.toDouble).collect()
      samples = samples ++ currentSamples
    }

    spark.sqlContext.createDataFrame(spark.sparkContext.parallelize(samples), df.schema)
  }

  // FIXME
  private def getSamplesToAdd(label: Double, sampleCount: Long, majorityClassCount: Int, samplingRatios: Map[Double, Double]): Int ={
    if(samplingRatios contains label) {
      val ratio = samplingRatios(label)
      if(ratio <= 1) {
        0
      } else {
        ((ratio - 1.0) * sampleCount).toInt
      }
    } else {
      majorityClassCount - sampleCount.toInt
    }
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    // FIXME - skip indexer for single class?
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

    val balanecedDF = if(counts.count() > 1) {
      val clsList = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)

      val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected.filter(datasetSelected($(labelCol))===clsList(x))))
        .map(x=>oversample(x._2, getSamplesToAdd(x._1.toDouble, x._2.count, majorityClassCount, $(samplingRatios))))

       datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    } else {
      oversample(datasetIndexed, $(singleClassOversamplingSize))
    }

    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol)).repartition(1)
  }


  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): RandomOversampleModel = {
    val copied = new RandomOversampleModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait RandomOversampleParams extends RandomOversampleModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class RandomOversample(override val uid: String) extends Estimator[RandomOversampleModel] with RandomOversampleParams {
  def this() = this(Identifiable.randomUID("randomOversample"))

  override def fit(dataset: Dataset[_]): RandomOversampleModel = {
    val model = new RandomOversampleModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): RandomOversample = defaultCopy(extra)

}




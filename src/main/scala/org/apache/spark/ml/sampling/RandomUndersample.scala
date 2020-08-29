package org.apache.spark.ml.sampling

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.knn.KNNModel
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import scala.collection.mutable
import scala.util.Random
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.{desc, udf}



/** Transformer Parameters*/
private[ml] trait RandomUndersampleModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class RandomUndersampleModel private[ml](override val uid: String) extends Model[RandomUndersampleModel] with RandomUndersampleModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  def underSample(df: DataFrame, numSamples: Int): DataFrame = {
    val spark = df.sparkSession
    var samples = Array[Row]() //FIXME - make this more parallel

    val underSampleRatio = numSamples / df.count().toDouble
    if (underSampleRatio < 1.0) {
      val currentSamples = df.sample(withReplacement = false, underSampleRatio, seed = 42L).collect()
      samples = samples ++ currentSamples
      val foo = spark.sparkContext.parallelize(samples)
      val x = spark.sqlContext.createDataFrame(foo, df.schema)
      x
    }
    else {
      df
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val counts = getCountsByClass(dataset.sparkSession, "label", dataset.toDF).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val labels = counts.collect().map(x=>x(0).toString.toInt)
    labels.filter(x=>x!=minClassLabel).map(x=>dataset.filter(dataset("label")===x)).map(x=>underSample(x.toDF, minClassCount)).reduce(_ union _)
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
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): RandomUndersampleModel = {
    val model = new RandomUndersampleModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): RandomUndersample = defaultCopy(extra)

}




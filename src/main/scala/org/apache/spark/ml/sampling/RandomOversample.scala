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
private[ml] trait RandomOversampleModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class RandomOversampleModel private[ml](override val uid: String) extends Model[RandomOversampleModel] with RandomOversampleModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  //assume there is only one class present
  def overSample(df: DataFrame, numSamples: Int): DataFrame = {
    println("oversample with " + numSamples)
    df.show
    var samples = Array[Row]() //FIXME - make this more parallel
    val spark = df.sparkSession
    //FIXME - some could be zero if split is too small
    //val samplesToAdd = numSamples - df.count()
    val currentCount = df.count()
    if (0 < currentCount && currentCount < numSamples) {
      val currentSamples = df.sample(withReplacement = true, (numSamples - currentCount) / currentCount.toDouble).collect()
      samples = samples ++ currentSamples
    }

    val foo = spark.sparkContext.parallelize(samples)
    val x = spark.sqlContext.createDataFrame(foo, df.schema)
    df.union(x).toDF()
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    val counts = getCountsByClass(dataset.sparkSession, "label", dataset.toDF).sort("_2")
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString.toInt
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt
    val labels = counts.collect().map(x=>x(0).toString.toInt)
    labels.filter(x=>x!=maxClassLabel).map(x=>dataset.filter(dataset("label")===x)).map(x=>overSample(x.toDF, maxClassCount)).reduce(_ union _)
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
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): RandomOversampleModel = {
    val model = new RandomOversampleModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): RandomOversample = defaultCopy(extra)

}




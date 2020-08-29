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
private[ml] trait ClassBalancerModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class ClassBalancerModel private[ml](override val uid: String) extends Model[ClassBalancerModel] with ClassBalancerModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))


  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.toDF()
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): ClassBalancerModel = {
    val copied = new ClassBalancerModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}




/** Estimator Parameters*/
private[ml] trait ClassBalancerParams extends ClassBalancerModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class ClassBalancer(override val uid: String) extends Estimator[ClassBalancerModel] with ClassBalancerParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): ClassBalancerModel = {
    val model = new ClassBalancerModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ClassBalancer = defaultCopy(extra)

}




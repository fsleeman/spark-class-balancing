package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import scala.collection.mutable
import scala.util.Random
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.desc


/** Transformer Parameters*/
private[ml] trait SMOTEModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class SMOTEModel private[ml](override val uid: String) extends Model[SMOTEModel] with SMOTEModelParams {
  def this() = this(Identifiable.randomUID("SMOTE"))

  def getSmoteSample(row: Row): Row = {
    val index = row(0).toString.toLong
    val label = row(1).toString.toInt
    val features: Array[Double] = row(2).asInstanceOf[DenseVector].toArray
    val neighbors = row(3).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail
    val randomNeighbor: Array[Double] = neighbors(Random.nextInt(neighbors.length)).toArray

    val gap = randomNeighbor.indices.map(_=>Random.nextDouble()).toArray // FIXME - should this be one value instead?

    val syntheticExample = Vectors.dense(Array(features, randomNeighbor, gap).transpose.map(x=>x(0) + x(2) * (x(1)-x(0)))).toDense

    Row(index, label, syntheticExample)
  }

  def oversample(df: Dataset[_], samplesToAdd: Int): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._

    val leafSize = 100
    val kValue = 5

    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(kValue + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f: KNNModel = model.fit(df)

    val t = f.transform(df).sort("index")
    println("*** first knn ****")
    t.show

    val dfCount = t.count.toInt
    //val randomIndicies = (0 until totalSamples - dfCount).map(_=>Random.nextInt(dfCount))
    val randomIndicies = (0 until samplesToAdd).map(_=>Random.nextInt(dfCount))
    val collected = t.withColumn("neighborFeatures", $"neighbors.features").drop("neighbors").collect
    val createdSamples = spark.createDataFrame(spark.sparkContext.parallelize(randomIndicies.map(x=>getSmoteSample(collected(x)))), df.schema).sort("index")

    createdSamples
  }

  override def transform(dataset: Dataset[_]): DataFrame= {
    println("~~~~~~~~~~~~~~~~~~~~~~~~smote")
    //val df = dataset.toDF()
    val counts = getCountsByClass(dataset.sparkSession, "label", dataset.toDF).sort("_2")
    counts.show
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val clsList: Array[Int] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toInt)
    //val clsDFs = clsList.indices.map(x=>oversample(df.filter(df("label")===clsList(x)), maxClassCount, union = true))
    //clsDFs.reduce(_ union _)

    val clsDFs = clsList.indices.map(x=>dataset.filter(dataset("label")===clsList(x))).map(x=>oversample(x, (majorityClassCount - x.count).toInt))   //    oversample(df.filter(df("label")===clsList(x)), maxClassCount, union = true))
    dataset.toDF.union(clsDFs.reduce(_ union _)) //fixme, not working the same between two methods
    //clsDFs.reduce(_ union _)
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

package org.apache.spark.ml.sampling

import com.sun.corba.se.impl.oa.toa.TransientObjectManager
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import scala.collection.mutable
import scala.util.Random

class smote {

  def getSmoteSample(row: Row): Row = {
    val index = row(0).toString.toLong
    val label = row(1).toString.toInt
    val features = row(2).asInstanceOf[DenseVector].toArray
    val neighbors = row(3).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail
    val randomNeighbor = neighbors(Random.nextInt(neighbors.length)).toArray

    val gap = randomNeighbor.indices.map(_=>Random.nextDouble()).toArray

    val syntheticExample = Vectors.dense(Array(features, randomNeighbor, gap).transpose.map(x=>x(0) + x(2) * (x(0)-x(1)))).toDense

    Row(index, label, syntheticExample)
  }

  def oversample(df: DataFrame, totalSamples: Int): DataFrame = {
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
    val randomIndicies = (0 until totalSamples - dfCount).map(_=>Random.nextInt(dfCount))
    val collected = t.withColumn("neighborFeatures", $"neighbors.features").drop("neighbors").collect
    df.union(spark.createDataFrame(spark.sparkContext.parallelize(randomIndicies.map(x=>getSmoteSample(collected(x)))), df.schema).sort("index"))
  }

  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {
    val df = dfIn
    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val clsList = counts.select("_1").filter(counts("_1") =!= maxClassLabel).collect().map(x=>x(0).toString.toInt)
    val clsDFs = clsList.indices.map(x=>oversample(df.filter(df("label")===clsList(x)), maxClassCount))
    clsDFs.reduce(_ union _)
  }
}
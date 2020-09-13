package org.apache.spark.ml.sampling

import breeze.stats.distributions.Gaussian
import org.apache.hadoop.hdfs.server.datanode.fsdataset.impl.RamDiskReplicaLruTracker
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random
// import org.apache.commons.math3.analysis.function.Gaussian




/** Transformer Parameters*/
private[ml] trait GaussianSMOTEModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class GaussianSMOTEModel private[ml](override val uid: String) extends Model[GaussianSMOTEModel] with GaussianSMOTEModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  def getSingleDistance(x: Array[Double], y: Array[Double]): Double = {
    var distance = 0.0
    for(index<-x.indices) {
      distance += (x(index) -  y(index)) *(x(index) - y(index))
    }
    distance
  }

  def getGaussian(u: Double, s: Double): Double = {
    val g = Gaussian(u, s)
    g.draw()
  }

  def getSmoteSample(row: Row): Row = {
    val index = row(0).toString.toLong
    val label = row(1).toString.toInt
    val features = row(2).asInstanceOf[DenseVector].toArray
    val neighbors = row(3).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail // skip the self neighbor
    val randomNeighbor = neighbors(Random.nextInt(neighbors.length)).toArray

    val randmPoint = Random.nextDouble()
    val gap = features.indices.map(x => (randomNeighbor(x) - features(x)) * randmPoint).toArray

    val sigma = 0.5 // FIXME - make it a parameter
    // FIXME - ask if this should be multi-dimensional?
    val ranges = features.indices.map(x => getGaussian(gap(x), sigma)).toArray
    val syntheticExample = Vectors.dense(Array(features, randomNeighbor, ranges).transpose.map(x => x(0) + (x(1) - x(0) * x(2)))).toDense

    Row(index, label, syntheticExample)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.filter((dataset("label") === 1) || (dataset("label") === 5)).toDF // FIXME
    val spark = df.sparkSession
    import spark.implicits._
    val m = 5   // k-value
    val leafSize = 1000

    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    /*** For each minority example, calculate the m nn's in training set***/
    val minorityDF = df.filter(df("label")===minClassLabel)
    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(m + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f = model.fit(minorityDF)
    val t = f.transform(minorityDF)

    t.show()
    t.printSchema()

    /*while len(samples) < num_to_sample:
        idx= self.random_state.randint(len(X_min))
        random_neighbor= self.random_state.choice(ind[idx][1:])
        s0 = self.sample_between_points(X_min[idx], X_min[random_neighbor])
        samples.append(self.random_state.normal(s0, self.sigma))*/

    val randomIndicies = (0 until samplesToAdd).map(_=>Random.nextInt(minorityDF.count.toInt))
    val collected = t.withColumn("neighborFeatures", $"neighbors.features").drop("neighbors").collect
    val createdSamples = spark.createDataFrame(spark.sparkContext.parallelize(randomIndicies.map(x=>getSmoteSample(collected(x)))), df.schema).sort("index")

    df.union(createdSamples)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): GaussianSMOTEModel = {
    val copied = new GaussianSMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}




/** Estimator Parameters*/
private[ml] trait GaussianSMOTEParams extends GaussianSMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class GaussianSMOTE(override val uid: String) extends Estimator[GaussianSMOTEModel] with GaussianSMOTEParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): GaussianSMOTEModel = {
    val model = new GaussianSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): GaussianSMOTE = defaultCopy(extra)

}

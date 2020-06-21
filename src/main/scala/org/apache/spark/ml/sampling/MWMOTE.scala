package org.apache.spark.ml.sampling

import com.sun.corba.se.impl.oa.toa.TransientObjectManager
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.sampling.utils.{getCountsByClass, getMatchingClassCount}
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import scala.collection.mutable
import scala.util.Random
import org.apache.spark.sql.functions._
import org.apache.spark.ml.sampling.utils.{pointDifference}

class MWNOTE {


  /*val getSafeNeighborCount = udf((array: mutable.WrappedArray[Int], minorityClassLabel: Int) => {
    def isMajorityNeighbor(x1: Int, x2: Int): Int = {
      if(x1 == x2) {
        1
      } else {
        0
      }
    }
    array.tail.map(x=>isMajorityNeighbor(minorityClassLabel, x)).sum
  })*/

  def explodeNeighbors(labels: mutable.WrappedArray[Int], features: mutable.WrappedArray[DenseVector]): Array[(Int, DenseVector)] = {
    val len = labels.length
    (1 until len).map(x=>(labels(x), features(x))).toArray
    //(0 until len).map(x=>features(x)).toArray
  }

  def calculateClosenessFactor(y: DenseVector, x: DenseVector, l: Int): Double ={
    val distance = pointDifference(y.toArray, x.toArray) / l.toFloat
    val CMAX = 2
    val Cf_th = 5

    val numerator = if(1/distance <= Cf_th) {
      1/distance
    } else {
      Cf_th
    }

    (numerator / Cf_th) * CMAX
  }

  def calculateDensityFactor(y: DenseVector, x: DenseVector): Double = {

    0.0
  }

  def calculateInformationWeight(y: DenseVector, x: DenseVector) : Double = {

    0.0
  }


  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {
    val df = dfIn
    import spark.implicits._

    val counts = getCountsByClass(spark, "label", df).sort("_2")
    counts.show()
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    /*1) find k1 NNs of minority examples
      2) from minority examples, remove examples with no minority example neighbors
      3) find k2 nearest majority examples for each in minority data
      4) union the nearest neighbor majority examples to build majority borderline set
      5) for each borderline majority example, find the k3 nearest neighbors from the minority examples
      6) union the nearest neighbors minority examples to build minority borderline set
      7) for majority/minority sets, compute the information weight
      8) for borderline minority, computer the selection weight
      9)  convert each to selection probability
      10) find clusters of minority examples
      11) initialize set
      12) for 1 to N
        a) selection x from borderline minority, by probability distribution
        b) selection sample y from cluster
        c) generate one synthetic example
        d) add example
    */


    println(minClassLabel, minClassCount)
    println(maxClassLabel, maxClassCount)

    val minorityDF = df.filter(df("label") === minClassLabel)
    val majorityDF = df.filter(df("label") =!= minClassLabel)

    val leafSize = 100
    /** 1 **/
    val k1 = 5
    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(k1 + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f1: KNNModel = model.fit(df)

    val Smin = f1.transform(minorityDF).sort("index")
    println("*** first knn ****")
    Smin.show

    /** 2 **/
    val Sminf = Smin.withColumn("nnMinorityCount", getMatchingClassCount($"neighbors.label", $"label")).sort("index").filter("nnMinorityCount > 0")
    Sminf.show

    /** 3 **/
    val k2 = 5
    val model2 = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(k2 + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f2: KNNModel = model.fit(majorityDF)

    val Nmaj = f2.transform(Sminf.drop("neighbors", "nnMinorityCount")).sort("index")
    println("*** NMaj ****")
    Nmaj.show

    /** 4 **/
    Nmaj.printSchema()
    val explodedNeighbors = Nmaj.select("neighbors").withColumn("label", $"neighbors.label")
      .withColumn("features", $"neighbors.features")
      .collect.flatMap(x=>explodeNeighbors(x(1).asInstanceOf[mutable.WrappedArray[Int]], x(2).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val Sbmaj = spark.sparkContext.parallelize(explodedNeighbors).toDF("label", "features").distinct()
    println(Sbmaj.count())

    /** 5 **/
    val k3 = 5
    val model3 = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(k3 + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f3: KNNModel = model.fit(majorityDF)

    val Nmin = f3.transform(Sminf.drop("neighbors", "nnMinorityCount")).sort("index")
    Nmin.show
    /** 6 **/
    val explodedNeighbors2 = Nmin.select("neighbors").withColumn("label", $"neighbors.label")
      .withColumn("features", $"neighbors.features")
      .collect.flatMap(x=>explodeNeighbors(x(1).asInstanceOf[mutable.WrappedArray[Int]], x(2).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val Simin = spark.sparkContext.parallelize(explodedNeighbors).toDF("label", "features").distinct()
    println(Sbmaj.count())

    /** 7a **/





    df
  }
}
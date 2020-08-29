package org.apache.spark.ml.sampling

import com.sun.corba.se.impl.oa.toa.TransientObjectManager
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.{getCountsByClass, getMatchingClassCount}
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

import scala.collection.mutable
import scala.util.Random
import org.apache.spark.sql.functions._
import org.apache.spark.ml.sampling.utils.pointDifference
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType


/** Transformer Parameters*/
private[ml] trait MWMOTEModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class MWMOTEModel private[ml](override val uid: String) extends Model[MWMOTEModel] with MWMOTEModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  def explodeNeighbors(labels: mutable.WrappedArray[Int], features: mutable.WrappedArray[DenseVector]): Array[(Int, DenseVector)] = {
    val len = labels.length
    (1 until len).map(x=>(labels(x), features(x))).toArray
    //(0 until len).map(x=>features(x)).toArray
  }
  
  def calculateDensityFactor(y: DenseVector, x: DenseVector): Double = {

    0.0
  }

  def calculateInformationWeight(y: DenseVector, x: DenseVector) : Double = {

    0.0
  }


  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF
    val spark = df.sparkSession
    import spark.implicits._

    val counts = getCountsByClass(spark, "label", df).sort("_2")
    counts.show()
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount

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

    val f2: KNNModel = model2.fit(majorityDF)

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
      .setTopTreeSize(minorityDF.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(k3 + 1) // include self example
      .setAuxCols(Array("label", "features"))

    /*val f3: KNNModel = model.fit(majorityDF)

    val Nmin = f3.transform(Sminf.drop("neighbors", "nnMinorityCount")).sort("index")*/
    println("minorityDF: " + minorityDF.count)
    val f3: KNNModel = model3.fit(minorityDF)
    val Nmin = f3.transform(Sbmaj.drop("neighbors", "nnMinorityCount"))//.sort("index")

    Nmin.show
    /** 6 **/
    val explodedNeighbors2 = Nmin.select("neighbors").withColumn("label", $"neighbors.label")
      .withColumn("features", $"neighbors.features")
      .collect.flatMap(x=>explodeNeighbors(x(1).asInstanceOf[mutable.WrappedArray[Int]], x(2).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    /*println(explodedNeighbors.length)
    for(x<-explodedNeighbors) {
      println(x)
    }*/
    // FIXME - correct Simin creation
    val Simin = spark.sparkContext.parallelize(explodedNeighbors2).toDF("label", "features").distinct()
    Simin.show
    //println(Sbmaj.count())

    /** 7 **/


    println("here")
    println(Sbmaj.count)


    val SiminCollected: Array[DenseVector] = Simin.select("features").collect().map(x=>x(0).asInstanceOf[DenseVector])

    val getClosenessFactors = udf((majorityExample: DenseVector) => {
      // FIXME - these values are always set to CMAX, wrong values for CMAX and Cf_th?
      def calculateClosenessFactor(y: DenseVector, x: DenseVector, l: Int): Double ={
        val distance = pointDifference(y.toArray, x.toArray) / l.toDouble
        val CMAX: Double = 3.0
        val Cf_th: Double = 5.0

        val numerator = if(1/distance <= Cf_th) {
          1/distance
        } else {
          Cf_th
        }
        (numerator / Cf_th) * CMAX
      }

      SiminCollected.map(x=>calculateClosenessFactor(majorityExample, x, majorityExample.size)).sorted
    })


    Sbmaj.show()
    val closenessFactors = Sbmaj.withColumn("closeness", getClosenessFactors(Sbmaj("features")))
    closenessFactors.show()
    closenessFactors.printSchema()

    val calculateInformationWeights = udf((closenessFactors: mutable.WrappedArray[Double]) => { // , closenessFactors: Array[Double]
      closenessFactors.toArray.map(x=>x*x/closenessFactors.toArray.sum)
    })

    val informationWeights = closenessFactors.withColumn("informationWeights", calculateInformationWeights(closenessFactors("closeness")))
    informationWeights.show

    informationWeights.select("informationWeights")
    /* 8 */ // FIXME - check this

    val calculateSelectionWeights = udf((informationWeights: mutable.WrappedArray[Double]) => { // , closenessFactors: Array[Double]
      informationWeights.toArray.sum
    })
    val selectionWeights = informationWeights.withColumn("selectionWeights", calculateSelectionWeights(informationWeights("informationWeights")))
    selectionWeights.show

    /* 9 */
    val selectionWeightSum = selectionWeights.select("selectionWeights").collect().map(x=>x(0).asInstanceOf[Double]).sum

    val calculateSelectionProbabilities = udf((selectionWeight: Double) => { // , closenessFactors: Array[Double]
      selectionWeight / selectionWeightSum
    })
    val selectionProbabilities = selectionWeights.withColumn("selectionProbabilities", calculateSelectionProbabilities(selectionWeights("selectionWeights")))
    selectionProbabilities.show

    // selection weights
    // selection probs
    // step 10
    // step 11
    // step 12

    // not sure what these were for
    /*val probs = selectionProbabilities.select("selectionProbabilities").collect.map(x=>x(0).asInstanceOf[Double])

    var probSums: Array[Double] = Array[Double](probs.length)
    for(i <- 1 until probs.length) {

    }*/

    /* 10 */
    val clusterKValue = 10
    val kmeans = new KMeans().setK(clusterKValue).setSeed(1L)
    val kMeansModel = kmeans.fit(Smin)
    val SiminKMeansClusters = kMeansModel.transform(Simin)

    SiminKMeansClusters.show

    /* 11 */
    // Somin = Smin

    /* 12 */

    // For all examples to generate:

    /* a */
    // Select a sample x from Simin according to probability distribution fS p ðx i Þg. Let, x is a member of the cluster Lk, 1 <= k <= M.
    // ** take example from Simin based on probability density
    // ** transform example, find cluster number

    def generateExample(): (Long, Int, DenseVector) = {
      // FIXME - need to redo collection/indexing of Simin

      // FIXME - change how this works with collected values
      val randomSamplesX = SiminKMeansClusters.filter(SiminKMeansClusters("prediction") === Random.nextInt(clusterKValue))
      println("count: " + randomSamplesX.count())

      val randomSamples = if(randomSamplesX.count() > 50) {
        randomSamplesX.sample(false, 0.1).take(2) // FIXME
      } else {
        randomSamplesX.sample(false, 1.0).take(2)
      }

      //println("take: " + randomSamples.length)
      val xSample = randomSamples(0)(1).asInstanceOf[DenseVector]
      //println(xSample)

      /* b */
      // Select another sample y, at random, from the members of the cluster Lk
      // ** pick random from cluster with previous cluster nunber
      val ySample = randomSamples(1)(1).asInstanceOf[DenseVector]
      // println(ySample)
      /* c */
      // Generate one synthetic data, s, according to s = x + a * (y - x), where a is a random number in range[0,1]
      // ** simple

      val alpha = Random.nextDouble()
      val synthetic = Array(xSample.toArray, ySample.toArray).transpose.map(x=> x(0) + alpha * (x(1) - x(0)))
      //println(synthetic)

      (0L, randomSamples(0)(0).toString.toInt, Vectors.dense(synthetic).toDense)

      /* d */
      // Add s to Somin : Somin = Somin U {s}
      // ** simple
    }

    // val syntheticExamples = (0 until samplesToAdd).map(_=>generateExample()) // FIXME - super slow, change method
    val syntheticExamples = (0 until 1).map(_=>generateExample())
    println("example count " + syntheticExamples.length)
    import spark.implicits._
    import df.sparkSession.implicits._
    val syntheticDF = spark.createDataFrame(spark.sparkContext.parallelize(syntheticExamples)).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2","label")
      .withColumnRenamed("_3","features").sort("index")

    df.show()
    syntheticDF.show

    df.printSchema()
    syntheticDF.printSchema()

    df.union(syntheticDF)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): MWMOTEModel = {
    val copied = new MWMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}




/** Estimator Parameters*/
private[ml] trait MWMOTEParams extends MWMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class MWMOTE(override val uid: String) extends Estimator[MWMOTEModel] with MWMOTEParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): MWMOTEModel = {
    val model = new MWMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): MWMOTE = defaultCopy(extra)

}

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
import org.apache.spark.sql.expressions.Window
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


  def oversampleClass(dataset: Dataset[_], minorityClassLabel: String, samplesToAdd: Int): DataFrame = {

    val df = dataset.toDF
    import df.sparkSession.implicits._
    val spark = df.sparkSession

    val Smin = df.filter(df("label") === minorityClassLabel)
    val Smaj = df.filter(df("label") =!= minorityClassLabel)

    //val minorityClassCount = Smin.count
    //val majorityClassCount = Smaj.count

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



    val leafSize = 100
    /** 1 **/
    val k1 = 5
    val model = new KNN().setFeaturesCol("features")
      //.setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeSize(10)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(k1 + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f1: KNNModel = model.fit(df)

    val SminNN = f1.transform(Smin).sort("index")
    println("*** first knn ****")
    SminNN.show

    /** 2 **/
    val Sminf = SminNN.withColumn("nnMinorityCount", getMatchingClassCount($"neighbors.label", $"label")).sort("index").filter("nnMinorityCount > 0")
    Sminf.show

    /** 3 **/
    val k2 = 5
    val model2 = new KNN().setFeaturesCol("features")
      //.setTopTreeSize(df.count().toInt / 1)   /// FIXME - check?
      .setTopTreeSize(10)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(k2) // Don't include self example
      .setAuxCols(Array("label", "features"))

    val f2: KNNModel = model2.fit(Smaj)

    val Nmaj = f2.transform(Sminf.drop("neighbors", "nnMinorityCount")).sort("index")
    println("*** NMaj ****")
    Nmaj.show

    /** 4 **/
    Nmaj.printSchema()
    val explodedNeighbors = Nmaj.select("neighbors").withColumn("label", $"neighbors.label")
      .withColumn("features", $"neighbors.features")
      .collect.flatMap(x=>explodeNeighbors(x(1).asInstanceOf[mutable.WrappedArray[Int]], x(2).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val Sbmaj = spark.sparkContext.parallelize(explodedNeighbors).toDF("label", "features").distinct()
    println("~~ Sbmaj")
    Sbmaj.show
    println(Sbmaj.count())

    /** 5 **/
    val k3 = 5
    val model3 = new KNN().setFeaturesCol("features")
      //.setTopTreeSize(Smin.count().toInt / 8)   /// FIXME - check?
      .setTopTreeSize(10)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(k3) // Don't include self example
      .setAuxCols(Array("label", "features"))

    /*val f3: KNNModel = model.fit(majorityDF)

    val Nmin = f3.transform(Sminf.drop("neighbors", "nnMinorityCount")).sort("index")*/
    println("minorityDF: " + Smin.count)
    val f3: KNNModel = model3.fit(Smin)
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
    println("~~ foo: ")
    Simin.show
    //println(Sbmaj.count())

    /** 7 **/


    println("here")
    println(Sbmaj.count)


    /*
      1) yi for each Sbmaj
      2) xi for each Simin
      3) Calculate Iw(yi, xi)
       a) Iw(yi, xi) = Cf(yi, xi) * Df(yi, xi)
       b) dn(yi, xi) = dist(yi, xi)/ l, l = number of dimensions
       c) Cf(yi, xi_) = Formula 6
       d) Df(yi, xi) = Formula 8

     */


    /*
    // ~~~~~~  By majority class
    val SiminCollected: Array[DenseVector] = Simin.select("features").collect().map(x=>x(0).asInstanceOf[DenseVector])

    val getClosenessFactors = udf((majorityExample: DenseVector) => {
      // FIXME - these values are always set to CMAX, wrong values for CMAX and Cf_th?
      def calculateClosenessFactor(y: DenseVector, x: DenseVector, l: Int): Double ={
        val distance = pointDifference(y.toArray, x.toArray) / l.toDouble
        val CMAX: Double = 3.0
        val Cf_th: Double = 50.0 // FIXME- should this be on the order of the number of features?

        val numerator = if(1/distance <= Cf_th) {
          1/distance
        } else {
          Cf_th
        }
        (numerator / Cf_th) * CMAX
      }

      SiminCollected.map(SiminValue=>calculateClosenessFactor(majorityExample, SiminValue, majorityExample.size)).sorted
    })


    Sbmaj.show()
    val closenessFactors = Sbmaj.withColumn("closeness", getClosenessFactors(Sbmaj("features"))) // result is majority data with minority closeness values
    closenessFactors.show()
    closenessFactors.printSchema()*/

    // ~~~~ by minority class
    val SbmajCollected: Array[DenseVector] = Sbmaj.select("features").collect().map(x=>x(0).asInstanceOf[DenseVector])

    val getClosenessFactors = udf((minorityExample: DenseVector) => {
      // FIXME - these values are always set to CMAX, wrong values for CMAX and Cf_th?
      def calculateClosenessFactor(x: DenseVector, y: DenseVector, l: Int): Double ={
        val distance = pointDifference(y.toArray, x.toArray) / l.toDouble
        val CMAX: Double = 3.0
        val Cf_th: Double = 50.0 // FIXME- should this be on the order of the number of features?

        val numerator = if(1/distance <= Cf_th) {
          1/distance
        } else {
          Cf_th
        }
        (numerator / Cf_th) * CMAX
      }

      SbmajCollected.map(SbmajValue=>calculateClosenessFactor(minorityExample, SbmajValue, 54)).sorted // FIXME - set feature length a member variable
    })


    // Sbmaj.show()
    val closenessFactors = Simin.withColumn("closeness", getClosenessFactors(Simin("features"))) // result is majority data with minority closeness values
    closenessFactors.show()
    closenessFactors.printSchema()


    val calculateInformationWeights = udf((closenessFactors: mutable.WrappedArray[Double]) => { // , closenessFactors: Array[Double]
      closenessFactors.toArray.map(x=>x*x/closenessFactors.toArray.sum)
    })

    val informationWeights = closenessFactors.withColumn("informationWeights", calculateInformationWeights(closenessFactors("closeness")))
    informationWeights.show

    // informationWeights.select("informationWeights") // this should be the information weights for each maj example, then for each minority examples



    /* 8 */ // FIXME - wrong

    val calculateSelectionWeights = udf((informationWeights: mutable.WrappedArray[Double]) => { // , closenessFactors: Array[Double]
      informationWeights.toArray.sum
    })
    val selectionWeights = informationWeights.withColumn("selectionWeights", calculateSelectionWeights(informationWeights("informationWeights")))
    selectionWeights.show

    println("selection weight count: " + selectionWeights.count() )

    /* 9 */
    val selectionWeightSum = selectionWeights.select("selectionWeights").collect().map(x=>x(0).asInstanceOf[Double]).sum

    val calculateSelectionProbabilities = udf((selectionWeight: Double) => { // , closenessFactors: Array[Double]
      selectionWeight / selectionWeightSum
    })
    val selectionProbabilities = selectionWeights.withColumn("selectionProbabilities", calculateSelectionProbabilities(selectionWeights("selectionWeights")))//.sort(col("selectionProbabilities").desc)
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
    val SiminKMeansClusters = kMeansModel.transform(selectionProbabilities)

    SiminKMeansClusters.show

    /* 11 */
    // Somin = Smin -- don't do it here


    /* 12 */

    val SiminClustersCollected = SiminKMeansClusters.select("label", "features", "selectionProbabilities", "prediction")
    SiminClustersCollected.show

    val partitionWindow = Window.partitionBy($"label").orderBy($"selectionProbabilities".desc).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    val sumTest = sum($"selectionProbabilities").over(partitionWindow)
    //val runningTotals = samplesToAddDF.select($"*", sumTest as "running_total")
    val runningTotals = SiminClustersCollected.select($"*", sumTest as "running_total")

    println("Running totals: ")
    runningTotals.show()

    val runningTotalsCollected = runningTotals.collect()

    val groupedClusters = (0 until clusterKValue).map(x=>runningTotalsCollected.filter(_(3)==x))


    val r = new scala.util.Random()

    def generateExample(): (Long, Int, DenseVector) = {

      val randomIndexPoint = r.nextDouble()
      val selectedRow = runningTotalsCollected.filter(x=>x(4).asInstanceOf[Double]>=randomIndexPoint)(0)// runningTotals.filter(runningTotals("running_total") >= randomIndexPoint).take(1)(0)

      val label = selectedRow(0).asInstanceOf[Int]
      val features = selectedRow(1).asInstanceOf[DenseVector]
      val cluster = selectedRow(3).asInstanceOf[Int]

      val randomIndex = (groupedClusters(cluster).length * r.nextDouble()).toInt
      val randomExample = groupedClusters(cluster)(randomIndex)(1).asInstanceOf[DenseVector]  // runningTotals.filter(runningTotals("prediction") === cluster).take(1)(0)(1).asInstanceOf[DenseVector]

      val alpha = r.nextDouble()
      val synthetic = Array(features.toArray, randomExample.toArray).transpose.map(x=> x(0) + alpha * (x(1) - x(0)))

      (0L, label, Vectors.dense(synthetic).toDense)
    }


    // val syntheticExamples = (0 until samplesToAdd).map(_=>generateExample()) // FIXME - super slow, change method
    val syntheticExamples = (0 until samplesToAdd).map(_=>generateExample())
    println("example count " + syntheticExamples.length)

    val syntheticDF = spark.createDataFrame(spark.sparkContext.parallelize(syntheticExamples)).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2","label")
      .withColumnRenamed("_3","features").sort("index")

    df.show()
    syntheticDF.show

    df.printSchema()
    syntheticDF.printSchema()

    syntheticDF
  }


  override def transform(dataset: Dataset[_]): DataFrame = {

    val df = dataset.toDF()
    // import df.sparkSession.implicits._

    val counts = getCountsByClass(df.sparkSession, "label", df).sort("_2")
    // val minClassLabel = counts.take(1)(0)(0).toString
    // val minClassCount = counts.take(1)(0)(1).toString.toInt
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt


    val minorityClasses = counts.collect.map(x=>(x(0).toString, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)
    val results: DataFrame = minorityClasses.map(x=>oversampleClass(dataset, x._1, majorityClassCount - x._2)).reduce(_ union _).union(dataset.toDF())

    println("dataset: " + dataset.count)
    println("added: " + results.count)

    results
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

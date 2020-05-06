package org.apache.spark.ml.sampling

import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.sampling.utils.{Element, getCountsByClass}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{Column, DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.mutable
import scala.util.Random

class adasyn {

  //val getMajorityNeighborCount2: UserDefinedFunction = udf((neighbors: mutable.WrappedArray[(Int, Element)]) => {
  /*val getMajorityNeighborCount2: UserDefinedFunction = udf((neighbors: Array[Element2]) => {
    val nearestClasses = neighbors.asInstanceOf[mutable.WrappedArray[Int]]
    val currentClass = nearestClasses(0)
    val majorityNeighbors = nearestClasses.tail.map(x=>if(x==currentClass) 0 else 1).sum
    //majorityNeighbors
    1
  })*/


  def generateExamples(row: Row): Array[Array[Double]] ={
    //println("*****")
    //println(row)
    val label = row(1).toString.toInt
    val examplesToCreate = row(5).asInstanceOf[Long].toInt
    val neighborLabels = row(6).asInstanceOf[mutable.WrappedArray[Int]]
    val neighborFeatures: mutable.Seq[DenseVector] = row(7).asInstanceOf[mutable.WrappedArray[DenseVector]]

    //var addedExamples = Array[Array[Double]]()
//
   // println("Adding " + examplesToCreate)
    if(neighborLabels.tail.contains(label)) {
     // println("found " + label)
      // skip self instance
      var minorityIndicies = Array[Int]()
      for(x<-1 until neighborLabels.length) {
        if(neighborLabels(x) == label ){
         // println(x, neighborLabels(x))
          minorityIndicies = minorityIndicies :+ x
        }
      }

      val randomIndicies = (0 until examplesToCreate).map(_=>minorityIndicies.toVector(Random.nextInt(minorityIndicies.size)))
      //println("*****")
      (0 until examplesToCreate).map(x=>neighborFeatures(randomIndicies(x)).toArray).toArray
    } else {
      //println("did not find " + label)
      //addedExamples = addedExamples ++ neighborFeatures.asInstanceOf[Array[Double]]
      val features: Array[Double] = neighborFeatures.head.toArray
      //println("*****")
      (0 until examplesToCreate).map(x=>features).toArray
    }

    /*val neighbors = row(3)
    val n2 = neighbors.asInstanceOf[Array[(Int, DenseVector)]]


    val x = row.toSeq(2).asInstanceOf[DenseVector].toArray
    println(x)*/



  }

  def fit(spark: SparkSession, dfIn: DataFrame, k: Int): DataFrame = {
    val df = dfIn
    //val spark = df.sparkSession
    import spark.implicits._

    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    println(minClassLabel, minClassCount)
    println(maxClassLabel, maxClassCount)

    val minorityDF = df.filter(df("label") === minClassLabel)
    val majorityDF = df.filter(df("label") =!= minClassLabel)

    val threshold = 1.0
    val beta = 1.0 // final balance level, might need to adjust from the original paper

    val imbalanceRatio = minClassCount.toDouble / maxClassCount.toDouble

    if (imbalanceRatio < threshold) {
      val G = (maxClassCount - minClassCount) * beta

      val leafSize = 100
      val kValue = 5
      val model = new KNN().setFeaturesCol("features")
        .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
        .setTopTreeLeafSize(leafSize)
        .setSubTreeLeafSize(leafSize)
        .setK(kValue + 1) // include self example
        .setAuxCols(Array("label", "features"))

      val f: KNNModel = model.fit(df)
      val t = f.transform(minorityDF)
      t.show

      val getMajorityNeighborRatio = udf((array: mutable.WrappedArray[Int]) => {
        def isMajorityNeighbor(x1: Int, x2: Int): Int = {
          if(x1 == x2) {
            0
          } else {
            1
          }
        }
        array.tail.map(x=>isMajorityNeighbor(array.head, x)).sum / kValue.toDouble
      })



      val collected = t.select($"neighbors.label")
      collected.show
      collected.printSchema()
      val dfNeighborRatio = t.withColumn("neighborClassRatio", getMajorityNeighborRatio($"neighbors.label"))//.drop("neighbors")
      dfNeighborRatio.show

      val neighborCountSum = dfNeighborRatio.agg(sum("neighborClassRatio")).first.get(0).toString.toDouble
      println(neighborCountSum)

      val getSampleCount = udf((density: Double) => {
        Math.round(density / neighborCountSum * G.toDouble)
      })

      val adjustedRatios = dfNeighborRatio.withColumn("densityDistribution", getSampleCount($"neighborClassRatio")).withColumn("labels", $"neighbors.label").withColumn("neighborFeatures", $"neighbors.features")
      adjustedRatios.show
      adjustedRatios.printSchema()

      val samplesToAddSum = adjustedRatios.agg(sum("densityDistribution")).first.get(0).toString.toDouble
      println("majority count: " + maxClassCount)
      println("minority count: " + minClassCount)
      println("samples to add: " + samplesToAddSum)  // FIXME - double check the size, wont be exactly the right number because of rounding

      adjustedRatios.withColumn("labels", $"neighbors.label").withColumn("neighborFeatures", $"neighbors.features").show

      val syntheticExamples: Array[Array[Array[Double]]] = adjustedRatios.collect.map(x=>generateExamples(x))

      println(syntheticExamples.length)
      val totalExamples = syntheticExamples.flatMap(x=>x.toSeq).map(x=>(0, minClassLabel.toInt, Vectors.dense(x).toDense))



      println("total: " + totalExamples.length)

      val bar = spark.createDataFrame(spark.sparkContext.parallelize(totalExamples))
      val bar2 = bar.withColumnRenamed("_1", "index")
        .withColumnRenamed("_2", "label")
        .withColumnRenamed("_3", "features")

      df.union(bar2)

    } else {
      df
    }
  }
}

package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.sampling.utils.pointDifference
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType

import scala.collection.mutable


/** Transformer Parameters*/
private[ml] trait SMOTEDModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class SMOTEDModel private[ml](override val uid: String) extends Model[SMOTEDModel] with SMOTEDModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  val knnK = 5

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.filter((dataset("label") === 1) || (dataset("label") === 5)).toDF // FIXME
    val spark = df.sparkSession
    val counts = getCountsByClass(spark, "label", df).sort("_2")
    import spark.implicits._
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    val minorityDF = df.filter(df("label") === minClassLabel)

    val leafSize = 10 // FIXME

    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(minorityDF.count().toInt / 8) /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(knnK + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f = model.fit(minorityDF)
    val neighbors = f.transform(minorityDF).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")
    neighbors.show()
    neighbors.printSchema()
    // udf for distances

    val calculateDistances = udf((neighbors: scala.collection.mutable.WrappedArray[DenseVector]) => {
      (1 until neighbors.length).map(x=>pointDifference(neighbors(0).toArray, neighbors(x).toArray))
    })

    val calculateStd = udf((distances: scala.collection.mutable.WrappedArray[Double]) => {
      val mean = distances.sum / distances.length.toDouble
      Math.sqrt(distances.map(x=>Math.pow(x - mean,2)).sum * (1 / (distances.length - 1).toDouble)) /// FIXME - check formula
    })

    val calculateLocalDistanceWeights = udf((distances: scala.collection.mutable.WrappedArray[Double]) => {
      distances.map(x=>x/distances.sum)
    })

    val calculateSamplesToAdd = udf((std: Double, distances: scala.collection.mutable.WrappedArray[Double]) => {
      //distances.indices.map(x=>(std * distances(x) * samplesToAdd))
      (std * minClassCount.toDouble).toInt
    })

    /*val sampleExistingExample =  udf((numberToAdd: Int, distances: scala.collection.mutable.WrappedArray[Double],
                                      neighbors: scala.collection.mutable.WrappedArray[DenseVector]) => {

      val counts = distances.map(x=>(((x + 0.5)/ distances.sum) * numberToAdd).toInt).reverse
      val originalExample = neighbors.head.toArray
      val reverseNeighbors = neighbors.tail.reverse.map(x=>x.toArray)

      // FIXME - only samples upto the required count



      def addOffset(x: Array[Double], distanceLine: Array[Double], offset: Double): Array[Double] = {
        Array[Array[Double]](x, distanceLine).transpose.map(x=>x(0) + x(1)*offset)
      }

      def getNeighborSamples(index: Int)= {
        val distanceLine: Array[Double] = Array[Array[Double]](originalExample, reverseNeighbors(index)).transpose.map(x=>x(1)-x(0))
        (0 until counts(index)).map(x=>addOffset(originalExample, distanceLine, x/counts(index).toDouble))
      }


      val x: Seq[IndexedSeq[Array[Double]]] = reverseNeighbors.indices.map(x=>getNeighborSamples(x))
      x.reduce(_ union _).toArray
    })*/

    def sampleExistingExample(numberToAdd: Int, distances: scala.collection.mutable.WrappedArray[Double],
                              neighbors: scala.collection.mutable.WrappedArray[DenseVector]) ={

      val counts = distances.map(x=>(((x + 0.5)/ distances.sum) * numberToAdd).toInt).reverse
      val originalExample = neighbors.head.toArray
      val reverseNeighbors = neighbors.tail.reverse.map(x=>x.toArray)

      // FIXME - only samples upto the required count



      def addOffset(x: Array[Double], distanceLine: Array[Double], offset: Double): Array[Double] = {
        Array[Array[Double]](x, distanceLine).transpose.map(x=>x(0) + x(1)*offset)
      }

      def getNeighborSamples(index: Int)= {
        val distanceLine: Array[Double] = Array[Array[Double]](originalExample, reverseNeighbors(index)).transpose.map(x=>x(1)-x(0))
        (0 until counts(index)).map(x=>addOffset(originalExample, distanceLine, x/counts(index).toDouble))
      }


      val x: Seq[IndexedSeq[Array[Double]]] = reverseNeighbors.indices.map(x=>getNeighborSamples(x))
      x.reduce(_ union _).toArray
    }


    val distances = neighbors.withColumn("distances", calculateDistances(neighbors("neighborFeatures")))
    distances.show

    val std = distances.withColumn("std", calculateStd(distances("distances")))
    std.show

    val stdSum = std.select("std").collect.map(x=>x(0).toString.toDouble).sum

    val stdWeights = std.withColumn("stdWeights", std("std")/stdSum)
    stdWeights.show

    val localDistanceWeights = stdWeights.withColumn("localDistanceWeights", calculateLocalDistanceWeights(stdWeights("distances")))
    localDistanceWeights.show

    val samplesToAddDF = localDistanceWeights.withColumn("samplesToAdd", calculateSamplesToAdd(localDistanceWeights("stdWeights"), localDistanceWeights("localDistanceWeights")))
    samplesToAddDF.show
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.expressions.Window
    import org.apache.spark.sql.functions._
    val calcSamplesToAdd = samplesToAddDF.select("samplesToAdd").collect.map(x=>x(0).toString.toDouble).sum

    val sortDF = samplesToAddDF.sort(col("samplesToAdd").desc)
    sortDF.show
    println(calcSamplesToAdd)

    val partitionWindow = Window.partitionBy($"label").orderBy($"samplesToAdd".desc)
    val sumTest = sum($"samplesToAdd").over(partitionWindow)
    val runningTotals = samplesToAddDF.select($"*", sumTest as "running_total")

    val filteredTotals = runningTotals.filter(runningTotals("running_total") < samplesToAdd)

    //val addedSamples: Array[Row] = filteredTotals.withColumn("added", sampleExistingExample(filteredTotals("samplesToAdd"), filteredTotals("distances"), filteredTotals("neighborFeatures"))).select("added").collect
    val addedSamples: Array[Array[Array[Double]]] = filteredTotals.collect.map(x=>sampleExistingExample(x(8).asInstanceOf[Int], x(4).asInstanceOf[mutable.WrappedArray[Double]], x(3).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val collectedSamples = addedSamples.reduce(_ union _).map(x=>Vectors.dense(x).toDense).map(x=>Row(0.toLong, minClassLabel.toInt, x.asInstanceOf[DenseVector]))

    println(collectedSamples.length)
    /*println(collectedSamples(0).length)
    for(x<-collectedSamples(0)) {
      print(x + " ")
    }
    println("")
    println(collectedSamples(1).length)
    println(collectedSamples(2).length)
*/

    val foo: Array[(Long, Int, DenseVector)] = collectedSamples.map(x=>x.toSeq).map(x=>(x.head.toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector]))
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))
    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    dataset.toDF.union(bar2)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): SMOTEDModel = {
    val copied = new SMOTEDModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait SMOTEDParams extends SMOTEDModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class SMOTED(override val uid: String) extends Estimator[SMOTEDModel] with SMOTEDParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): SMOTEDModel = {
    val model = new SMOTEDModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): SMOTED = defaultCopy(extra)

}

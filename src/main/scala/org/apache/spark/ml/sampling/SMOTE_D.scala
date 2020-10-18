package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, calculateToTreeSize, getSamplesToAdd}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.sampling.utils.pointDifference
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

import scala.collection.mutable


/** Transformer Parameters*/
private[ml] trait SMOTEDModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {

}


/** Transformer */
class SMOTEDModel private[ml](override val uid: String) extends Model[SMOTEDModel] with SMOTEDModelParams {
  def this() = this(Identifiable.randomUID("smoteD"))

  val knnK = 5

  private val calculateDistances: UserDefinedFunction = udf((neighbors: scala.collection.mutable.WrappedArray[DenseVector]) => {
    (1 until neighbors.length).map(x=>pointDifference(neighbors(0).toArray, neighbors(x).toArray))
  })

  private val calculateStd: UserDefinedFunction = udf((distances: scala.collection.mutable.WrappedArray[Double]) => {
    val mean = distances.sum / distances.length.toDouble
    Math.sqrt(distances.map(x=>Math.pow(x - mean,2)).sum * (1 / (distances.length - 1).toDouble)) /// FIXME - check formula
  })

  private val calculateLocalDistanceWeights: UserDefinedFunction = udf((distances: scala.collection.mutable.WrappedArray[Double]) => {
    distances.map(x=>x/distances.sum)
  })

  private def sampleExistingExample(numberToAdd: Int, distances: scala.collection.mutable.WrappedArray[Double],
                            neighbors: scala.collection.mutable.WrappedArray[DenseVector]): Array[Array[Double]] ={

    val counts = distances.map(x=>((x/ distances.sum) * numberToAdd).toInt + 1).reverse

    val originalExample = neighbors.head.toArray
    val reverseNeighbors = neighbors.tail.reverse.map(x=>x.toArray) // furtherest first

    // FIXME - only samples upto the required count


    def addOffset(x: Array[Double], distanceLine: Array[Double], offset: Double): Array[Double] = {
      Array[Array[Double]](x, distanceLine).transpose.map(x=>x(0) + x(1)*offset)
    }

    def getNeighborSamples(index: Int)= {
      val distanceLine: Array[Double] = Array[Array[Double]](originalExample, reverseNeighbors(index)).transpose.map(x=>x(1)-x(0))
      (0 until counts(index)).map(x=>addOffset(originalExample, distanceLine, (x + 1)/(counts(index) + 1).toDouble))
    }

    val x: Seq[IndexedSeq[Array[Double]]] = reverseNeighbors.indices.map(x=>getNeighborSamples(x))
    x.reduce(_ union _).toArray.take(numberToAdd)
  }


  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val df = dataset.toDF
    val spark = df.sparkSession

    import spark.implicits._

    val model = new KNN().setFeaturesCol($(featuresCol))
      // .setTopTreeSize($(topTreeSize))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), dataset.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val f = model.fit(dataset)
    val neighbors = f.transform(dataset).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")

    val distances = neighbors.withColumn("distances", calculateDistances(neighbors("neighborFeatures")))
    //distances.show

    val std = distances.withColumn("std", calculateStd(distances("distances")))
    ///std.show

    val stdSum = std.select("std").collect.map(x=>x(0).toString.toDouble).sum

    val stdWeights = std.withColumn("stdWeights", std("std")/stdSum)
    //stdWeights.show

    val localDistanceWeights = stdWeights.withColumn("localDistanceWeights", calculateLocalDistanceWeights(stdWeights("distances")))
    localDistanceWeights.show

    val calculateSamplesToAdd = udf((std: Double, distances: scala.collection.mutable.WrappedArray[Double]) => {
      (std * samplesToAdd).toInt + 1 // fix for too few values based on rounding
    })

    val samplesToAddDF = localDistanceWeights.withColumn("samplesToAdd", calculateSamplesToAdd(localDistanceWeights("stdWeights"), localDistanceWeights("localDistanceWeights")))
    println("samples to add: " + samplesToAdd)

    val sortDF = samplesToAddDF.sort(col("stdWeights").desc)

    val partitionWindow = Window.partitionBy(col($(labelCol))).orderBy($"samplesToAdd".desc).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    val sumTest = sum($"samplesToAdd").over(partitionWindow)
    val runningTotals = sortDF.select($"*", sumTest as "running_total")

    val filteredTotals = runningTotals.filter(runningTotals("running_total") <= samplesToAdd) // FIXME - use take if less then amount
    println("samples to add: " + samplesToAdd)
    println("~~~~~" + filteredTotals.count)
    filteredTotals.printSchema()

    val addedSamples: Array[Array[Array[Double]]] = filteredTotals.collect.map(x=>sampleExistingExample(x(7).asInstanceOf[Int], x(3).asInstanceOf[mutable.WrappedArray[Double]], x(2).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    println("addedSamples: " + addedSamples.length)
    val collectedSamples = addedSamples.reduce(_ union _).map(x=>Vectors.dense(x).toDense).map(x=>Row(minorityClassLabel, x))

    // println(collectedSamples.length)

    spark.createDataFrame(dataset.sparkSession.sparkContext.parallelize(collectedSamples), dataset.schema)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol($(labelCol))
      .setOutputCol("labelIndexed")

    val datasetIndexed = indexer.fit(dataset).transform(dataset)
      .withColumnRenamed($(labelCol), "originalLabel")
      .withColumnRenamed("labelIndexed",  $(labelCol))
    datasetIndexed.show()
    datasetIndexed.printSchema()

    val labelMap = datasetIndexed.select("originalLabel",  $(labelCol)).distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
    val labelMapReversed = labelMap.map(x=>(x._2, x._1))

    val datasetSelected = datasetIndexed.select($(labelCol), $(featuresCol))
    val counts = getCountsByClass(datasetSelected.sparkSession, $(labelCol), datasetSelected.toDF).sort("_2")
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)

    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected.filter(datasetSelected($(labelCol))===clsList(x))))
      .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, x._2.count, majorityClassCount, $(samplingRatios))))

    val balanecedDF = datasetIndexed.select($(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol)).repartition(1)
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
  def this() = this(Identifiable.randomUID("smoteD"))

  override def fit(dataset: Dataset[_]): SMOTEDModel = {
    val model = new SMOTEDModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): SMOTED = defaultCopy(extra)

}

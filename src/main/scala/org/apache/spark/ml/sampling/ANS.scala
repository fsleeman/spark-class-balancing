package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCol, HasSeed}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random


/** Transformer Parameters*/
private[ml] trait ANSModelParams extends Params with HasFeaturesCol with HasInputCol {
  /**
    * Param for if the nearest neighbor should be based on distance, not k
    * Default: False
    *
    * @group param
    */
  val samplingRatios = new Param[Map[Double, Double]](this, "samplingRatios", "map of sampling ratios per class")

  /** @group getParam */
  def getSamplingRatios: Map[Double, Double] = $(samplingRatios)

  def setSamplingRatios(value: Map[Double, Double]): this.type = set(samplingRatios, value)


  setDefault(samplingRatios -> Map(), inputCol->"label")
}

/** Transformer */
class ANSModel private[ml](override val uid: String) extends Model[ANSModel] with ANSModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))


  def createSample(row: Row): Array[Row] = {
    val index = row(0).toString.toLong
    val label = row(1).toString.toInt
    val features: Array[Double] = row(2).asInstanceOf[DenseVector].toArray
    val neighbors = row(3).asInstanceOf[mutable.WrappedArray[DenseVector]].toArray.tail
    val samplesToAdd = row(4).toString.toInt

    def addSample(): Row ={
      println("neighbor count: " + neighbors.length)
      val randomNeighbor: Array[Double] = neighbors(Random.nextInt(neighbors.length)).toArray
      val gap = Random.nextDouble()
      val syntheticExample = Vectors.dense(Array(features, randomNeighbor).transpose.map(x=>x(0) + gap * (x(1)-x(0)))).toDense
      Row(index, label, syntheticExample)
    }

    (0 until samplesToAdd).map(x=>addSample()).toArray
  }

  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    //val df = dataset.toDF // .filter((dataset("label") === 5) || (dataset("label") === 6)).toDF // FIXME
    val datasetSelected = dataset.select("index", "label", "features")
    //val spark = df.sparkSession
    import datasetSelected.sparkSession.implicits._
    //val counts = getCountsByClass(spark, "label", datasetSelected).sort("_2")
    //val minClassLabel = counts.take(1)(0)(0).toString
    //val minClassCount = counts.take(1)(0)(1).toString.toInt
    //val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    //val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    //val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    val minorityDF = datasetSelected.filter(datasetSelected("label") === minorityClassLabel)
    val majorityDF = datasetSelected.filter(datasetSelected("label") =!= minorityClassLabel)

    val C_max = Math.ceil(0.25 * datasetSelected.count()).toInt

    val leafSize = 10 // FIXME

    val minorityKnnModel: KNN = new KNN().setFeaturesCol("features")
      .setTopTreeSize(10) /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(1 + 1) // include self example
      .setAuxCols(Array("label", "features")).setBalanceThreshold(0.0) // FIXME

    val getNearestNeighborDistance = udf((distances: mutable.WrappedArray[Double]) => {
      distances(1)
    })

    val minorityKnnFit: KNNModel = minorityKnnModel.fit(minorityDF).setDistanceCol("distances")

    val neighborDistances = minorityKnnFit.transform(minorityDF).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")

    println("---> firstPosNeighborDistances (distance col)")
    neighborDistances.show()
    neighborDistances.printSchema()

    val firstPosNeighborDistances = neighborDistances.withColumn("closestPosDistance", getNearestNeighborDistance($"distances")).drop("distances", "neighborFeatures")
    println("---> firstPosNeighborDistances (closestPosDistance col)")
    firstPosNeighborDistances.show


    val majorityKnnModel: KNN = new KNN().setFeaturesCol("features")
      .setTopTreeSize(10) /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      //.setK(20) // include self example
      .setAuxCols(Array("label", "features")).setBalanceThreshold(0.0) // FIXME
    //.setQueryByDistance(true)   // FIXME - move this

    // println("@@ query mode: " + majorityKnnModel.getQueryMode)

    val majorityKnnFit: KNNModel = majorityKnnModel.fit(majorityDF).setDistanceCol("distances").setMaxDistanceCol("closestPosDistance").setQueryByDistance(true)//.setK(20)

    //val majorityNeighbors = majorityKnnFit.transform(minorityClosestDistance.filter(minorityClosestDistance("index")===9670)).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")
    val majorityNeighbors = majorityKnnFit.transform(firstPosNeighborDistances).withColumn("neighborFeatures", $"neighbors.features").drop("neighbors")
    println("---> majorityNeighbors")
    majorityNeighbors.show()
    majorityNeighbors.printSchema()

    val getRadiusNeighbors = udf((distances: mutable.WrappedArray[Double]) => {
      distances.length
    })


    val outBorder = majorityNeighbors.withColumn("outBorder", getRadiusNeighbors($"distances"))
    println("---> outBorder (outBorder col)")
    outBorder.show

    val outBorderArray = outBorder.select("outBorder").collect().map(x => x(0).asInstanceOf[Int])
    println("outborder " + outBorderArray.length)
    println("max:" + outBorderArray.max)

    var previous_number_of_outcasts = -1
    var C = 1
    //var best_diff = Int.MaxValue

    import scala.util.control._
    val loop = new Breaks
    loop.breakable {
      for (c <- 1000 until C_max) { // FIXME

        val number_of_outcasts = outBorderArray.filter(x => x >= c).sum
        println("loop " + c + " " + number_of_outcasts + " " + previous_number_of_outcasts)

        if (Math.abs(number_of_outcasts - previous_number_of_outcasts) == 0) {
          C = c
          if(outBorder.filter(outBorder("outBorder") < C).count > 0) {
            loop.break()
          }
        }
        previous_number_of_outcasts = number_of_outcasts
      }
    }


    println("C_max: " + C_max)
    println("C: " + C)
    //val OC = outBorder.filter(outBorder("outBorder") >= C)
    //OC.show
    //println("OC count: " + OC.count)
    val Pused = outBorder.filter(outBorder("outBorder") < C).drop("distances", "neighborFeatures", "outBorder")
    println("Pused count: " + Pused.count)
    Pused.show


    val PusedKnnModel: KNN = new KNN().setFeaturesCol("features")
      .setTopTreeSize(2) /// FIXME - wont work with small sample sizes
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      //.setK(1 + 1) // include self example
      .setAuxCols(Array("label", "features")).setBalanceThreshold(0.0) // FIXME

    val PusedKnnFit: KNNModel = PusedKnnModel.fit(Pused).setDistanceCol("distances").setMaxDistanceCol("closestPosDistance").setQueryByDistance(true)
    val PusedDistances = PusedKnnFit.transform(Pused).withColumn("neighborFeatures", $"neighbors.features")
      .drop("neighbors").withColumn("neighborCount", getRadiusNeighbors($"distances"))
    println("---> PusedDistances")
    PusedDistances.show


    val neighborCountSum = PusedDistances.select("neighborCount").collect().map(x=>x(0).toString.toInt).sum.toDouble

    println("neighborCountSum " + neighborCountSum)

    val getSamplesToAdd = udf((count: Int) => {
      Math.ceil((count / neighborCountSum) * samplesToAdd).toInt
    })

    val generatedSampleCounts = PusedDistances.withColumn("samplesToAdd", getSamplesToAdd($"neighborCount"))

    generatedSampleCounts.show

    val syntheticExamples: Array[Array[Row]] = generatedSampleCounts.drop("closestPosDistance", "distances", "neighborCount").filter("neighborCount > 0")
      .collect.map(x=>createSample(x))

    val totalExamples: Array[Row] = syntheticExamples.flatMap(x => x.toSeq).take(samplesToAdd) // FIXME

    val bar = datasetSelected.sparkSession.createDataFrame(datasetSelected.sparkSession.sparkContext.parallelize(totalExamples), datasetSelected.schema)
    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    // df.union(bar2)
    bar2
  }


    override def transform(dataset: Dataset[_]): DataFrame = {
      val datasetSelected = dataset.select("index", "label", "features")

      val counts = getCountsByClass(datasetSelected.sparkSession, "label", datasetSelected).sort("_2")
      val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
      val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

      val clsList: Array[String] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString)//.take(2)

      println("********* " + clsList(0))


      def getSamplesToAdd(label: Double, sampleCount: Long): Int ={
        if($(samplingRatios) contains label) {
          val ratio = $(samplingRatios)(label)
          if(ratio <= 1) {
            0
          } else {
            ((ratio - 1.0) * sampleCount).toInt
          }
        } else {
          majorityClassCount - sampleCount.toInt
        }
      }

      val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected.filter(datasetSelected("label")===clsList(x))))
        .map(x=>oversample(datasetSelected, x._1.toDouble, getSamplesToAdd(x._1.toDouble, x._2.count)))
      datasetSelected.toDF.union(clsDFs.reduce(_ union _))
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): ANSModel = {
    val copied = new ANSModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait ANSParams extends ANSModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class ANS(override val uid: String) extends Estimator[ANSModel] with ANSParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): ANSModel = {
    val model = new ANSModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ANS = defaultCopy(extra)

}


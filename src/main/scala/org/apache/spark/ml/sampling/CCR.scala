package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, getSamplesToAdd}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, desc, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable


/** Transformer Parameters*/
private[ml] trait CCRModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {

}

/** Transformer */
class CCRModel private[ml](override val uid: String) extends Model[CCRModel] with CCRModelParams {
  def this() = this(Identifiable.randomUID("ccr"))

  type Element = (Int, Array[Double])
  type Element2 = (Long, Int, Array[Double])


  def getManhattanDistance(example: Array[Double], neighbor: Array[Double]): Double ={
    Array(example, neighbor).transpose.map(x=>Math.abs(x(1)-x(0))).sum
  }

  val moveMajorityPoints2: UserDefinedFunction = udf((features: DenseVector, neighborIndices: mutable.WrappedArray[Long],
                                                      neighborLabels: mutable.WrappedArray[Double], neighborFeatures: mutable.WrappedArray[DenseVector], distanceArray: mutable.WrappedArray[Double], ri: Double) => {
    val majorityIndicies: Array[Long] = neighborIndices.toArray
    val majorityLabels: Array[Double] = neighborLabels.toArray
    val majorityNeighbors: Array[Array[Double]] = neighborFeatures.toArray.map(x=>x.toArray)

    val distances = distanceArray.toArray

    def pointDistance(features: Array[Double], neighbor: Array[Double]): Double ={
      Array(features, neighbor).transpose.map(x=>Math.abs(x(1) - x(0))).sum
    }

    // type MajorityPoint = (Long, Int, Array[Double])

    def getMovedNeighbors(j: Int): (Boolean, (Long, Double, Array[Double])) ={
      if(distances(j) <= ri) {
        val d = pointDistance(features.toArray, majorityNeighbors(j))
        // FIXME - check line 19 in algorithm for tj usage (ask about this)
        // FIXME - devide by zero val scale = (ri - d) / d
        val scale = if(d == 0) {
          ri
        } else {
          (ri - d) / d
        }
        // println(scale)
        val offset: Array[Double] = Array(features.toArray, majorityNeighbors(j)).transpose.map(x=>x(1) - x(0)).map(x=>x * scale)
        val updatedPosition = Array(offset, majorityNeighbors(j)).transpose.map(x=>x(0)+x(1))
        (true, (majorityIndicies(j), majorityLabels(j), updatedPosition))
      } else {
        (false, (majorityIndicies(j), majorityLabels(j), majorityNeighbors(j)))
      }
    }

    majorityNeighbors.indices.map(j=>getMovedNeighbors(j)).filter(x=>x._1).map(x=>x._2)
  })

  private def NoP(distances: Array[Double], radius: Double): Int = {
    def isWithinRadius(d: Double): Int ={
      if (d <= radius) {
        1
      } else {
        0
      }
    }

    distances.map(x=>isWithinRadius(x)).sum + 1
  }

  private def nearestNoWithinR(distances: Array[Double], r: Double): Double ={

    def setWithinValue(d: Double, r: Double): Double ={
      if(d < r) {
        // FIXME - check this is max value could happen (added fix)
        Double.MaxValue
      } else {
        d
      }
    }
    distances.map(x=>setWithinValue(x, r)).min
  }

  private val stuff: UserDefinedFunction = udf((distanceArray: mutable.WrappedArray[Double]) => {

    val distances = distanceArray.toArray
    var energyBudget = 0.64
    var ri = 0.0
    var deltaR = energyBudget

    // generate cleaning radius
    while(energyBudget > 0.0) {
      val NoPValue = NoP(distances, ri)
      deltaR = energyBudget / NoPValue.toDouble
      if(NoP(distances, ri + deltaR) > NoPValue) {
        deltaR = distances.filter(x=>x>ri).min
      }
      if(deltaR == Double.MaxValue) {
        energyBudget = 0.0
      } else {
        ri = ri + deltaR
        energyBudget = energyBudget - deltaR * NoPValue
      }
    }
    // Math.pow(ri, -1)
    ri
  })

  private def extractMovedPoints(index: Array[Long], label: Array[Double], feature: Array[Array[Double]]): Array[Row] ={
    index.indices.map(x=>Row(index(x), label(x), feature(x))).toArray
  }

  private def createSyntheicPoints(row: Row): Array[Row] ={
    val label = row(0).toString
    val features = row(1).asInstanceOf[DenseVector].toArray
    val r = row(2).toString.toDouble
    val examplesToAdd = Math.ceil(row(3).toString.toDouble).toInt // FIXME - check this

    val random = scala.util.Random
    // (0 until examplesToAdd).map(_=>Row(0L, label, for(f <- features) yield Vectors.dense(f * (random.nextDouble() * 2.0 - 1) * r))).toArray
    (0 until examplesToAdd).map(_=>Row(0L, label, Vectors.dense(for(f <- features) yield f * (random.nextDouble() * 2.0 - 1) * r))).toArray
  }

  private def getAveragedRow(rows: Array[Row]): Row ={
    val data: Array[Double] = rows.map(x=>x(2).asInstanceOf[Array[Double]]).transpose.map(x=>x.sum).map(x=>x/rows.length.toDouble)
    Row(rows(0)(0).toString.toLong, rows(0)(1).toString.toDouble, Vectors.dense(data))
  }

  private def oversample(dataset: DataFrame, minorityClassLabel: Double, samplesToAdd: Int): DataFrame ={
    // parameters
    // proportion = 1.0, energy = 1.0, scaling = 0.0 // FIXME - add parameters
    val spark = dataset.sparkSession
    import spark.implicits._

    println("at dataset")
    dataset.show

    println("Samples to add: " + samplesToAdd)
    println("dataset count: " + dataset.count)
    println("minorityClassLabel: " + minorityClassLabel)

    val minorityDF = dataset.filter(dataset($(labelCol)) === minorityClassLabel)
    val majorityDF = dataset.filter(dataset($(labelCol)) =!= minorityClassLabel)
    println("minority count: " + minorityDF.count)
    minorityDF.show


    /// FIXME - switch to distance?
    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize($(topTreeSize))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array("index", $(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))



    println("majorityDF count: " + majorityDF.count)


    val fitModel: KNNModel = model.fit(majorityDF)
    fitModel.setDistanceCol("distances")

    val minorityDataNeighbors = fitModel.transform(minorityDF)//.sort("index")
    println("*** first knn ****")
    minorityDataNeighbors.show
    minorityDataNeighbors.printSchema()


    val test = minorityDataNeighbors.withColumn("majorityIndex", $"neighbors.index")
      .withColumn("majorityLabel", $"neighbors.label")
      .withColumn("majorityPoints", $"neighbors.features").drop("neighbors")//.take(1)
    test.show
    test.printSchema()

    // FIXME - use full DF
    val test2 = test.collect() // test.take(10)

    val foo = test2.map(x=>x.toSeq).map(x=>(x(0).toString.toLong, x(1).toString, x(2).asInstanceOf[DenseVector],
      x(3).asInstanceOf[mutable.WrappedArray[Double]], x(4).asInstanceOf[mutable.WrappedArray[Long]],
      x(5).asInstanceOf[mutable.WrappedArray[Double]], x(6).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))
    val testDF = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", $(labelCol))
      .withColumnRenamed("_3", $(featuresCol))
      .withColumnRenamed("_4", "distances")
      .withColumnRenamed("_5", "majorityIndex")
      .withColumnRenamed("_6", "majorityLabel")
      .withColumnRenamed("_7", "majorityFeatures")

    testDF.show
    testDF.printSchema()

    val result = testDF.withColumn("ri", stuff($"distances"))
    result.show

    val invertRi: UserDefinedFunction = udf((ri: Double) => {
      Math.pow(ri, -1)
    })

    val inverseRi = result.withColumn("riInverted", invertRi($"ri"))
    val inverseRiSum = inverseRi.select("riInverted").rdd.map(x=>x(0).toString.toDouble).reduce(_ + _)
    println("inverse sum " + inverseRiSum)

    //val resultWithSampleCount = inverseRi.withColumn("gi", ($"riInverted"/ inverseRiSum) * samplesToAdd).sort(col("gi").desc)
    //resultWithSampleCount.show

    val resultWithSampleCount = inverseRi.withColumn("gi", $"riInverted"/ inverseRiSum)//.sort(col("gi").desc)
    resultWithSampleCount.show

    val giSum = resultWithSampleCount.select("gi").rdd.map(x=>x(0).toString.toDouble).reduce(_ + _)
    println(giSum)

    val resulsWithSamplesToAdd = resultWithSampleCount.withColumn("samplesToAdd", ($"gi"/ giSum) * samplesToAdd).sort(col("samplesToAdd").desc)
    resulsWithSamplesToAdd.show

    // XXXXX
    // FIXME - should the sampling rate be proportional of gi?
    val createdPoints: Array[Array[Row]] = resulsWithSamplesToAdd.drop("index", "distances", "majorityIndex",
      "majorityLabel", "majorityFeatures", "riInverted", "gi").collect().map(x=>createSyntheicPoints(x))

    println("created points length: " + createdPoints.length)


    val unionedPoints = createdPoints.reduce(_ union _).take(samplesToAdd)

    println("~~~~~~ oversampled points: " + unionedPoints.length)

    val movedPoints = resultWithSampleCount.withColumn("movedMajorityPoints",
      moveMajorityPoints2($"features",  $"majorityIndex",  $"majorityLabel", $"majorityFeatures", $"distances", $"ri"))
    movedPoints.show()
    movedPoints.printSchema()
    println("moved points: " + movedPoints.count())

    // XXXXXX

    val movedPointsExpanded = movedPoints.withColumn("movedMajorityIndex", $"movedMajorityPoints._1")
      .withColumn("movedMajorityLabel", $"movedMajorityPoints._2")
      .withColumn("movedMajorityExamples", $"movedMajorityPoints._3")
      .drop("movedMajorityPoints")


    val movedPointsSelected = movedPointsExpanded.select("movedMajorityIndex", "movedMajorityLabel", "movedMajorityExamples")
    movedPointsSelected.show()




    val movedPointsCollected = movedPointsSelected.collect()

    val fooX: Array[(Array[Long], Array[Double], Array[Array[Double]])] = movedPointsCollected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[Long]].toArray,
      x(1).asInstanceOf[mutable.WrappedArray[Double]].toArray,
      x(2).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Double]]].toArray.map(y=>y.toArray)))

    val results = fooX.map(x=>extractMovedPoints(x._1, x._2, x._3))

    val total: Array[Row] = results.reduce(_ union _)
    println(total.length)

    val grouped: Map[Long, Array[Row]] = total groupBy (s => s(0).toString.toLong)

    val averaged: Array[Row] = grouped.map(x=>getAveragedRow(x._2)).toArray
    val movedMajorityIndicies = averaged.map(x=>x(0).toString.toLong).toList

    val movedMajorityExamplesDF = spark.createDataFrame(spark.sparkContext.parallelize(averaged.map(x=>(x(0).toString.toLong, x(1).toString.toDouble, x(2).asInstanceOf[DenseVector])))).toDF() // x(2).asInstanceOf[Array[Double]])))).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2",$(labelCol))
      .withColumnRenamed("_3",$(featuresCol))

    //movedMajorityExamplesDF.show
    //movedMajorityExamplesDF.printSchema()
    //println(movedMajorityExamplesDF.count())

    val syntheticExamplesDF = spark.createDataFrame(spark.sparkContext.parallelize(unionedPoints.map(x=>(x(0).toString.toLong, x(1).toString.toDouble, x(2).asInstanceOf[DenseVector])))).toDF() // x(2).asInstanceOf[Array[Double]])))).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2",$(labelCol))
      .withColumnRenamed("_3",$(featuresCol))

    //syntheticExamplesDF.show
    //syntheticExamplesDF.printSchema()
    //println(syntheticExamplesDF.count)

    val keptMajorityDF = dataset.filter(!$"index".isin(movedMajorityIndicies: _*))
    //println(keptMajorityDF.count)
    keptMajorityDF.show()
    //keptMajorityDF.printSchema()
    //println(keptMajorityDF.count())

    val finalDF = keptMajorityDF.union(movedMajorityExamplesDF).union(syntheticExamplesDF)//.drop("index")

    finalDF.show()
    println(finalDF.count())

    finalDF
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

    val datasetSelected = datasetIndexed.select("index", $(labelCol), $(featuresCol))
    datasetSelected.printSchema()

    println("~~~~~~~~~~~~~~~~~~~")
    val counts = getCountsByClass(datasetSelected.sparkSession, $(labelCol), datasetSelected.toDF).sort("_2")
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString.toDouble
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    // val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)
    val minorityClasses = counts.collect.map(x=>(x(0).toString.toDouble, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)//.sortBy(_._2)//.reverse

    //val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected))
    //  .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, x._2.count, majorityClassCount, $(samplingRatios))))

    var ds = datasetSelected




    for(minorityClass<-minorityClasses) {
      ds = oversample(ds, minorityClass._1, majorityClassCount - minorityClass._2)
    }



    //val balanecedDF = datasetIndexed.select($(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    // FIXME - test run did not generate new samples, adjust parameters
    ds.drop("index").withColumn("originalLabel", restoreLabel(ds.col($(labelCol)))).drop($(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol)).repartition(1)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): CCRModel = {
    val copied = new CCRModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait CCRParams extends CCRModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class CCR(override val uid: String) extends Estimator[CCRModel] with CCRParams {
  def this() = this(Identifiable.randomUID("ccr"))

  override def fit(dataset: Dataset[_]): CCRModel = {
    val model = new CCRModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): CCR = defaultCopy(extra)

}


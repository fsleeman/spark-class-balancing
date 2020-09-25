package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, desc, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable






/** Transformer Parameters*/
private[ml] trait CCRModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class CCRModel private[ml](override val uid: String) extends Model[CCRModel] with CCRModelParams {
  def this() = this(Identifiable.randomUID("ccr"))

  type Element = (Int, Array[Double])
  type Element2 = (Long, Int, Array[Double])


  def getManhattanDistance(example: Array[Double], neighbor: Array[Double]): Double ={
    Array(example, neighbor).transpose.map(x=>Math.abs(x(0)-x(1))).sum
  }

  val moveMajorityPoints2: UserDefinedFunction = udf((features: DenseVector, neighborIndices: mutable.WrappedArray[Long],
                                                      neighborLabels: mutable.WrappedArray[Int], neighborFeatures: mutable.WrappedArray[DenseVector], distanceArray: mutable.WrappedArray[Double], ri: Double) => {

    val majorityIndicies: Array[Long] = neighborIndices.toArray
    val majorityLabels: Array[Int] = neighborLabels.toArray
    val majorityNeighbors: Array[Array[Double]] = neighborFeatures.toArray.map(x=>x.toArray)

    // val neighborDistances: Array[Double] = distanceArray.toArray
    val distances = distanceArray.toArray

    def pointDistance(features: Array[Double], neighbor: Array[Double]): Double ={
      Array(features, neighbor).transpose.map(x=>Math.abs(x(0) - x(1))).sum
    }

    type MajorityPoint = (Long, Int, Array[Double])

    def getMovedNeighbors(j: Int): (Boolean, (Long, Int, Array[Double])) ={
      // println("^^^ " + distances(j) + " " + ri)
      if(distances(j) <= ri) {
        val d = pointDistance(features.toArray, majorityNeighbors(j))
        // FIXME - check line 19 in algorithm for tj usage (ask about this)
        val scale =  (ri - d) / d
        val offset: Array[Double] = Array(features.toArray, majorityNeighbors(j)).transpose.map(x=>x(1) - x(0)).map(x=>x * scale)
        val updatedPosition = Array(offset, majorityNeighbors(j)).transpose.map(x=>x(0)+x(1))
        (true, (majorityIndicies(j), majorityLabels(j), updatedPosition))
      } else {
        (false, (majorityIndicies(j), majorityLabels(j), majorityNeighbors(j)))
      }
    }

    // println("~~~ indicies: " + majorityNeighbors.indices.length)
    val movedMajoirtyNeigbors = majorityNeighbors.indices.map(j=>getMovedNeighbors(j)).filter(x=>x._1).map(x=>x._2)
    movedMajoirtyNeigbors
  })

  /*val moveMajorityPoints: UserDefinedFunction = udf((features: DenseVector, neighborIndices: mutable.WrappedArray[Long],
                                                     neighborLabels: mutable.WrappedArray[Int], neighborFeatures: mutable.WrappedArray[DenseVector], distanceArray: mutable.WrappedArray[Double], ri: Double) => {
    val majorityIndicies: Array[Long] = neighborIndices.toArray
    val majorityLabels: Array[Int] = neighborLabels.toArray
    val majorityNeighbors: Array[Array[Double]] = neighborFeatures.toArray.map(x=>x.toArray)

    // val neighborDistances: Array[Double] = distanceArray.toArray
    val distances = distanceArray.toArray

    def pointDistance(features: Array[Double], neighbor: Array[Double]): Double ={
      Array(features, neighbor).transpose.map(x=>Math.abs(x(0) - x(1))).sum
    }

    type MajorityPoint = (Long, Int, Array[Double])

    def getMovedNeighbors(j: Int): Array[(Long, Int, Array[Double])] ={
      // println("^^^ " + distances(j) + " " + ri)
      if(distances(j) <= ri) {
        // println("@@", distances(j), ri)
        val d = pointDistance(features.toArray, majorityNeighbors(j))
        // FIXME - check line 19 in algorithm for tj usage
        val scale =  (ri - d) / d
        val offset: Array[Double] = Array(features.toArray, majorityNeighbors(j)).transpose.map(x=>x(0) - x(1)).map(x=>x * scale)
        val updatedPosition = Array(offset, majorityNeighbors(j)).transpose.map(x=>x(0)+x(1))

        Array[(Long, Int, Array[Double])]((majorityIndicies(j), majorityLabels(j), updatedPosition))
      } else {
        Array[(Long, Int, Array[Double])]()
      }
    }

    println("~~~ indicies: " + majorityNeighbors.indices.length)
    val movedMajoirtyNeigbors = majorityNeighbors.indices.map(j=>getMovedNeighbors(j))
    val combinedMajoritNeighbors: Array[(Long, Int, Array[Double])] = movedMajoirtyNeigbors.reduce(_ union _)
    print(combinedMajoritNeighbors(0))
    print("***** " + combinedMajoritNeighbors.length)

    combinedMajoritNeighbors
    // return cleaning radius and moved majority points

  })*/

  def NoP(distances: Array[Double], radius: Double): Int = {
    def isWithinRadius(d: Double): Int ={
      if (d <= radius) {
        1
      } else {
        0
      }
    }

    distances.map(x=>isWithinRadius(x)).sum + 1
  }

  def nearestNoWithinR(distances: Array[Double], r: Double): Double ={

    def setWithinValue(d: Double, r: Double): Double ={
      if(d < r) {
        // println("123456") // FIXME - check this is max value could happen (added fix)
        Double.MaxValue
      } else {
        d
      }
    }
    distances.map(x=>setWithinValue(x, r)).min
  }

  val stuff: UserDefinedFunction = udf((distanceArray: mutable.WrappedArray[Double]) => {

    val distances = distanceArray.toArray
    var energyBudget = 0.64
    var ri = 0.0
    var deltaR = energyBudget

    // generate cleaning radius
    while(energyBudget > 0.0) {
      val NoPValue = NoP(distances, ri)
      deltaR = energyBudget / NoPValue.toDouble
      if(NoP(distances, ri + deltaR) > NoPValue) {
        // deltaR = nearestNoWithinR(distances, ri)
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


  def extractMovedPoints(index: Array[Long], label: Array[Int], feature: Array[Array[Double]]): Array[Row] ={
    index.indices.map(x=>Row(index(x), label(x), feature(x))).toArray
  }

  def createSyntheicPoints(row: Row): Array[Row] ={
    //println(row.toString())
    val label = row(0).toString
    val features = row(1).asInstanceOf[DenseVector].toArray
    val r = row(2).toString.toDouble
    val examplesToAdd = Math.ceil(row(3).toString.toDouble).toInt // FIXME - check this

    val random = scala.util.Random
    // (0 until examplesToAdd).map(_=>Row(0L, label, for(f <- features) yield Vectors.dense(f * (random.nextDouble() * 2.0 - 1) * r))).toArray
    (0 until examplesToAdd).map(_=>Row(0L, label, Vectors.dense(for(f <- features) yield f * (random.nextDouble() * 2.0 - 1) * r))).toArray
  }


  //def oversampleClass(dataset: Dataset[_], minorityClassLabel: String, samplesToAdd: Int): DataFrame ={
  def oversampleClass(df: DataFrame, minorityClassLabel: String, samplesToAdd: Int): DataFrame ={
    // parameters
    // proportion = 1.0, energy = 1.0, scaling = 0.0
    // val df = dataset.toDF
    //val df = dataset.filter((dataset("label") === 1) || (dataset("label") === 5)).toDF // FIXME
    val spark = df.sparkSession
    import spark.implicits._

    //val counts = getCountsByClass(spark, "label", df).sort("_2")
    //val minClassLabel = counts.take(1)(0)(0).toString
    //val minClassCount = counts.take(1)(0)(1).toString.toInt
    //val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    //val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    //val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    val minorityDF = df.filter(df("label") === minorityClassLabel)
    val majorityDF = df.filter(df("label") =!= minorityClassLabel)

    val leafSize = 100
    val kValue = 10 /// FIXME - switch to distance?

    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(kValue + 1) // include self example
      .setAuxCols(Array("index", "label", "features"))
      //.setQueryByDistance(false)

    //println(model.getQueryMode)

    val f: KNNModel = model.fit(majorityDF)
    f.setDistanceCol("distances")

    val t = f.transform(minorityDF).sort("index")
    println("*** first knn ****")
    t.show
    t.printSchema()


    val test = t.withColumn("majorityIndex", $"neighbors.index")
      .withColumn("majorityLabel", $"neighbors.label")
      .withColumn("majorityPoints", $"neighbors.features").drop("neighbors")//.take(1)
    test.show
    test.printSchema()

    // FIXME - use full DF
    val test2 = test.collect()// test.take(10)

    val foo = test2.map(x=>x.toSeq).map(x=>(x.head.toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector],
      x(3).asInstanceOf[mutable.WrappedArray[Double]], x(4).asInstanceOf[mutable.WrappedArray[Long]],
      x(5).asInstanceOf[mutable.WrappedArray[Int]], x(6).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))
    val testDF = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")
      .withColumnRenamed("_4", "distances")
      .withColumnRenamed("_5", "majorityIndex")
      .withColumnRenamed("_6", "majorityLabel")
      .withColumnRenamed("_7", "majorityFeatures")


    //println("****** at result")
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

    // FIXME - should the sampling rate be proportional of gi?

    val createdPoints: Array[Array[Row]] = resulsWithSamplesToAdd.drop("index", "distances", "majorityIndex",
      "majorityLabel", "majorityFeatures", "riInverted", "gi").collect().map(x=>createSyntheicPoints(x))
    println("***")
    /*for(x<-createdPoints) {
      println(x.length)
    }*/
    println("created points length: " + createdPoints.length)

    val unionedPoints = createdPoints.reduce(_ union _).take(samplesToAdd)
    println("~~~~~~ oversampled points: " + unionedPoints.length)

    val movedPoints = resultWithSampleCount.withColumn("movedMajorityPoints",
      moveMajorityPoints2($"features",  $"majorityIndex",  $"majorityLabel", $"majorityFeatures", $"distances", $"ri"))
    movedPoints.show()
    movedPoints.printSchema()

    val movedPointsExpanded = movedPoints.withColumn("movedMajorityIndex", $"movedMajorityPoints._1")
      .withColumn("movedMajorityLabel", $"movedMajorityPoints._2")
      .withColumn("movedMajorityExamples", $"movedMajorityPoints._3")
      .drop("movedMajorityPoints")


    val movedPointsSelected = movedPointsExpanded.select("movedMajorityIndex", "movedMajorityLabel", "movedMajorityExamples")
    movedPointsSelected.show()

    val movedPointsCollected = movedPointsSelected.collect()

    val fooX: Array[(Array[Long], Array[Int], Array[Array[Double]])] = movedPointsCollected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[Long]].toArray,
      x(1).asInstanceOf[mutable.WrappedArray[Int]].toArray,
      x(2).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Double]]].toArray.map(y=>y.toArray)))

    val results = fooX.map(x=>extractMovedPoints(x._1, x._2, x._3))

    val total: Array[Row] = results.reduce(_ union _)
    println(total.length)

    val grouped: Map[Long, Array[Row]] = total groupBy (s => s(0).toString.toLong)


    def getAveragedRow(rows: Array[Row]): Row ={
      val data: Array[Double] = rows.map(x=>x(2).asInstanceOf[Array[Double]]).transpose.map(x=>x.sum).map(x=>x/rows.length.toDouble)
      Row(rows(0)(0).toString.toLong, rows(0)(1).toString.toInt, Vectors.dense(data))
    }

    val averaged: Array[Row] = grouped.map(x=>getAveragedRow(x._2)).toArray

    val movedMajorityIndicies = averaged.map(x=>x(0).toString.toLong).toList


    println("############")
    println(averaged(0))
    println(unionedPoints(0))


    val movedMajorityExamplesDF = spark.createDataFrame(spark.sparkContext.parallelize(averaged.map(x=>(x(0).toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector])))).toDF() // x(2).asInstanceOf[Array[Double]])))).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2","label")
      .withColumnRenamed("_3","features")

    movedMajorityExamplesDF.show
    movedMajorityExamplesDF.printSchema()
    println(movedMajorityExamplesDF.count())


    val syntheticExamplesDF = spark.createDataFrame(spark.sparkContext.parallelize(unionedPoints.map(x=>(x(0).toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector])))).toDF() // x(2).asInstanceOf[Array[Double]])))).toDF()
      .withColumnRenamed("_1","index")
      .withColumnRenamed("_2","label")
      .withColumnRenamed("_3","features")

    syntheticExamplesDF.show
    syntheticExamplesDF.printSchema()
    println(syntheticExamplesDF.count)



    val keptMajorityDF = df.filter(!$"index".isin(movedMajorityIndicies: _*))
    println(keptMajorityDF.count)
    keptMajorityDF.show()
    keptMajorityDF.printSchema()
    println(keptMajorityDF.count())


    val finalDF = keptMajorityDF.union(movedMajorityExamplesDF).union(syntheticExamplesDF)

    finalDF.show()
    println(finalDF.count())

    finalDF
  }



  override def transform(dataset: Dataset[_]): DataFrame = {

    var df = dataset.toDF()

    val counts = getCountsByClass(df.sparkSession, "label", df).sort("_2")
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    //val minorityClasses = counts.collect.map(x=>(x(0).toString, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)
   //  val results: DataFrame = minorityClasses.map(x=>oversampleClass(dataset, x._1, majorityClassCount - x._2)).reduce(_ union _).union(dataset.toDF())

    // val results: DataFrame = minorityClasses.map(x=>oversampleClass(dataset, x._1, majorityClassCount - x._2)).reduce(_ union _).union(dataset.toDF())

    val minorityClasses = counts.collect.map(x=>(x(0).toString, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)//.sortBy(_._2)//.reverse
    for(x<-minorityClasses) {
      println(x._1, x._2)
    }

    // FIXME - does the order matter?
    for(minorityClass<-minorityClasses) {
      df = oversampleClass(df, minorityClass._1, majorityClassCount - minorityClass._2)
    }

    println("dataset: " + dataset.count)
    println("added: " + df.count)

    df
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


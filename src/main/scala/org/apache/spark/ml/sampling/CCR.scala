package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.Utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, calculateToTreeSize, getSamplesToAdd}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, desc, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}


import scala.collection.mutable


/** Transformer Parameters*/
private[ml] trait CCRModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {
  /**
    * Param for energy.
    * @group param
    */
  final val energy: Param[Double] = new Param[Double](this, "energy", "energy")

  /** @group getParam */
  final def setEnergy(value: Double): this.type = set(energy, value)

  /**
    * Param for kNN top tree leaf size.
    * @group param
    */
  final val distanceNeighborLimit: Param[Int] = new Param[Int](this, "distanceNeighborLimit", "limit of how many distance based neighbors to keep, default=100, using 0 will returns all")

  /** @group getParam */
  final def setdDstanceNeighborLimit(value: Int): this.type = set(distanceNeighborLimit, value)

  setDefault(energy -> 1.0, distanceNeighborLimit -> 100)
}

/** Transformer */
class CCRModel private[ml](override val uid: String) extends Model[CCRModel] with CCRModelParams {
  def this() = this(Identifiable.randomUID("ccr"))

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

    def getMovedNeighbors(j: Int): (Boolean, (Long, Double, Array[Double])) ={
      if(distances(j) <= ri) {
        val d = pointDistance(features.toArray, majorityNeighbors(j))
        // FIXME - check line 19 in algorithm for tj usage (ask about this)
        val scale = if(d == 0) {
          ri
        } else {
          (ri - d) / d
        }

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
        // FIXME - check this if max value could happen (added fix)
        Double.MaxValue
      } else {
        d
      }
    }
    distances.map(x=>setWithinValue(x, r)).min
  }

  private val stuff: UserDefinedFunction = udf((distanceArray: mutable.WrappedArray[Double]) => {

    val distances = distanceArray.toArray
    var energyBudget = $(energy)
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
    ri
  })

  private def extractMovedPoints(index: Array[Long], label: Array[Double], feature: Array[Array[Double]]): Array[Row] ={
    index.indices.map(x=>Row(index(x), label(x), feature(x))).toArray
  }

  private def createSyntheicPoints(row: Row): Array[Row] ={
    val label = row(0).toString
    val features = row(1).asInstanceOf[DenseVector].toArray
    val r = row(2).toString.toDouble
    val examplesToAdd = Math.ceil(row(3).toString.toDouble).toInt

    val random = scala.util.Random
    (0 until examplesToAdd).map(_=>Row(0L, label, Vectors.dense(for(f <- features) yield f * (random.nextDouble() * 2.0 - 1) * r))).toArray
  }

  private def getAveragedRow(rows: Array[Row]): Row ={
    val data: Array[Double] = rows.map(x=>x(2).asInstanceOf[Array[Double]]).transpose.map(x=>x.sum).map(x=>x/rows.length.toDouble)
    Row(rows(0)(0).toString.toLong, rows(0)(1).toString.toDouble, Vectors.dense(data))
  }

  private def oversample(dataset: DataFrame, minorityClassLabel: Double, samplesToAdd: Int): DataFrame ={

    val spark = dataset.sparkSession
    import spark.implicits._

    val minorityDF = dataset.filter(dataset($(labelCol)) === minorityClassLabel)
    val majorityDF = dataset.filter(dataset($(labelCol)) =!= minorityClassLabel)

    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), majorityDF.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setAuxCols(Array("index", $(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel: KNNModel = model.fit(majorityDF)
    fitModel.setDistanceCol("distances").setQueryByDistance(true).setDistanceNeighborLimit($(distanceNeighborLimit))

    val minorityDataNeighbors = fitModel.transform(minorityDF)

    val minorityDataNeighborsReshaped: Array[Row] = minorityDataNeighbors.withColumn("majorityIndex", $"neighbors.index")
      .withColumn("majorityLabel", $"neighbors.label")
      .withColumn("majorityPoints", $"neighbors.features").drop("neighbors").collect()

    val minorityDataNeighborsArray = minorityDataNeighborsReshaped.map(x=>x.toSeq).map(x=>(x(0).toString.toLong, x(1).toString, x(2).asInstanceOf[DenseVector],
      x(3).asInstanceOf[mutable.WrappedArray[Double]], x(4).asInstanceOf[mutable.WrappedArray[Long]],
      x(5).asInstanceOf[mutable.WrappedArray[Double]], x(6).asInstanceOf[mutable.WrappedArray[DenseVector]]))
    val minorityDataNeighborsDF = spark.createDataFrame(spark.sparkContext.parallelize(minorityDataNeighborsArray))
      .withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", $(labelCol))
      .withColumnRenamed("_3", $(featuresCol))
      .withColumnRenamed("_4", "distances")
      .withColumnRenamed("_5", "majorityIndex")
      .withColumnRenamed("_6", "majorityLabel")
      .withColumnRenamed("_7", "majorityFeatures")

    val result = minorityDataNeighborsDF.withColumn("ri", stuff($"distances"))

    val invertRi: UserDefinedFunction = udf((ri: Double) => {
      Math.pow(ri, -1)
    })

    val inverseRi = result.withColumn("riInverted", invertRi($"ri"))

    val X: RDD[Double] = inverseRi.select("riInverted").rdd.map(x=>x(0).toString.toDouble)


    if(X.count() == 0) { // random oversample
      val samples = dataset.sample(withReplacement = true, (dataset.count + samplesToAdd) / dataset.count.toDouble)
      samples
    } else {
      val inverseRiSum = X.reduce(_ + _)

      val resultWithSampleCount = inverseRi.withColumn("gi", $"riInverted" / inverseRiSum)

      val giSum = resultWithSampleCount.select("gi").rdd.map(x => x(0).toString.toDouble).reduce(_ + _)

      val resulsWithSamplesToAdd = resultWithSampleCount.withColumn("samplesToAdd", ($"gi" / giSum) * samplesToAdd).sort(col("samplesToAdd").desc)

      // FIXME - should the sampling rate be proportional of gi?
      val createdPoints: Array[Array[Row]] = resulsWithSamplesToAdd.drop("index", "distances", "majorityIndex",
        "majorityLabel", "majorityFeatures", "riInverted", "gi").collect().map(x => createSyntheicPoints(x))

      val unionedPoints = createdPoints.reduce(_ union _).take(samplesToAdd)

      val movedPoints = resultWithSampleCount.withColumn("movedMajorityPoints",
        moveMajorityPoints2($"features", $"majorityIndex", $"majorityLabel", $"majorityFeatures", $"distances", $"ri"))

      val movedPointsExpanded = movedPoints.withColumn("movedMajorityIndex", $"movedMajorityPoints._1")
        .withColumn("movedMajorityLabel", $"movedMajorityPoints._2")
        .withColumn("movedMajorityExamples", $"movedMajorityPoints._3")
        .drop("movedMajorityPoints")

      val movedPointsSelected = movedPointsExpanded.select("movedMajorityIndex", "movedMajorityLabel", "movedMajorityExamples")
      val movedPointsCollected = movedPointsSelected.collect()

      val fooX: Array[(Array[Long], Array[Double], Array[Array[Double]])] = movedPointsCollected.map(x => (x(0).asInstanceOf[mutable.WrappedArray[Long]].toArray,
        x(1).asInstanceOf[mutable.WrappedArray[Double]].toArray,
        x(2).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Double]]].toArray.map(y => y.toArray)))

      val results = fooX.map(x => extractMovedPoints(x._1, x._2, x._3))

      val total: Array[Row] = results.reduce(_ union _)

      val grouped: Map[Long, Array[Row]] = total groupBy (s => s(0).toString.toLong)

      val averaged: Array[Row] = grouped.map(x => getAveragedRow(x._2)).toArray
      val movedMajorityIndicies = averaged.map(x => x(0).toString.toLong).toList

      val movedMajorityExamplesDF = spark.createDataFrame(spark.sparkContext.parallelize(averaged.map(x => (x(0).toString.toLong, x(1).toString.toDouble, x(2).asInstanceOf[DenseVector])))).toDF()
        .withColumnRenamed("_1", "index")
        .withColumnRenamed("_2", $(labelCol))
        .withColumnRenamed("_3", $(featuresCol))

      val syntheticExamplesDF = spark.createDataFrame(spark.sparkContext.parallelize(unionedPoints.map(x => (x(0).toString.toLong, x(1).toString.toDouble, x(2).asInstanceOf[DenseVector])))).toDF()
        .withColumnRenamed("_1", "index")
        .withColumnRenamed("_2", $(labelCol))
        .withColumnRenamed("_3", $(featuresCol))

      val keptMajorityDF = dataset.filter(!$"index".isin(movedMajorityIndicies: _*))
      val finalDF = keptMajorityDF.union(movedMajorityExamplesDF).union(syntheticExamplesDF)

      finalDF
    }
  }

  val checkForNegatives: UserDefinedFunction = udf((features: DenseVector) => {
    if(features.values.min < 0.0 || features.values.count(_.isNaN) > 0) {
      true
    } else {
      false
      }
    })

  def removeNegatives(df: DataFrame): DataFrame ={
    val negatives = df.withColumn("negativesPresent", checkForNegatives(df.col($(featuresCol))))
    negatives.filter(negatives("negativesPresent")=!=true).drop("negativesPresent")
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val indexer = new StringIndexer()
      .setInputCol($(labelCol))
      .setOutputCol("labelIndexed")

    val datasetIndexed = indexer.fit(dataset).transform(dataset)
      .withColumnRenamed($(labelCol), "originalLabel")
      .withColumnRenamed("labelIndexed",  $(labelCol))

    val labelMap = datasetIndexed.select("originalLabel",  $(labelCol)).distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
    val labelMapReversed = labelMap.map(x=>(x._2, x._1))

    val datasetSelected = datasetIndexed.select("index", $(labelCol), $(featuresCol))

    val counts = getCountsByClass(datasetSelected.sparkSession, $(labelCol), datasetSelected.toDF).sort("_2")
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString.toDouble
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val minorityClasses = counts.collect.map(x=>(x(0).toString.toDouble, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)

    var ds = datasetSelected
    for(minorityClass<-minorityClasses) {
      ds = oversample(ds, minorityClass._1, majorityClassCount - minorityClass._2)
    }
    ds.show()
    ds = removeNegatives(ds)

    val restoreLabel = udf((label: Double) => labelMapReversed(label))
    ds.drop("index").withColumn("originalLabel", restoreLabel(ds.col($(labelCol)))).drop($(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))//.repartition(1)
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


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
import org.apache.spark.sql.functions.{desc, udf, lit}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType




/** Transformer Parameters*/
private[ml] trait SafeLevelSMOTEModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {

}

/** Transformer */
class SafeLevelSMOTEModel private[ml](override val uid: String) extends Model[SafeLevelSMOTEModel] with SafeLevelSMOTEModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  private val getSafeNeighborCount = udf((array: mutable.WrappedArray[Double], minorityClassLabel: Double) => {
    def isMajorityNeighbor(x1: Double, x2: Double): Int = {
      if(x1 == x2) {
        1
      } else {
        0
      }
    }
    array.tail.map(x=>isMajorityNeighbor(minorityClassLabel, x)).sum
  })


  private val getRandomNeighbor = udf((labels: mutable.WrappedArray[Double], features: mutable.WrappedArray[DenseVector], rnd: Double) => {
    val randomIndex = (rnd * labels.length-1).toInt + 1
    //Vector(labels(randomIndex), features(randomIndex))
    (labels(randomIndex), features(randomIndex))
  })

  private val randUdf = udf({() => Random.nextDouble()})
  private val randFeatureUdf = udf({array: DenseVector =>
    Vectors.dense(Array.fill(array.size)(Random.nextDouble())).toDense
  })

  private val generateExample2 = udf((pLabel: Double, nLabel: Double, pFeatures: DenseVector, nFeatures:DenseVector, pLabelCount: Int, nLabelCount: Int, rnds2: DenseVector) => {
    val rnds: Array[Double] = rnds2.toArray

    val slRatio = if(nLabelCount != 0) {
      pLabelCount / nLabelCount.toDouble
    }
    else {
      Double.NaN
    }
    val featureCount = pFeatures.size
    val gap = if(slRatio == 1) {
      //Array.fill[Double](featureCount)(Random.nextDouble())
      rnds
    } else if(slRatio > 1) {
      // 0 to 1/slRatio
      Array(Array.fill[Double](featureCount)(1 / slRatio), rnds).transpose.map(x=>x(0) * x(1))
    } else if(slRatio < 1) {
      //  1-slRatio
      //Array.fill[Double](featureCount)((1-slRatio) + (1-slRatio) * Random.nextDouble())
      Array(Array.fill[Double](featureCount)((1-slRatio) + (1-slRatio)), rnds).transpose.map(x=>x(0) * x(1))
    } else {
      Array.fill[Double](featureCount)(0.0) // <-- if(slRatio == Double.NaN  )
    }

    val data: Array[Array[Double]] = Array(pFeatures.toArray, nFeatures.toArray, rnds)
    //data.map(x=>x(0) + (x(0)-x(1))*x(2))
    //data.map(x=>x(0))
    pFeatures.toArray
  })

  /*private val getRatios = udf({(pClassCount: Int, nClassCounts: Array[Int] )=>

    def getRatio(nClassCount: Int): Double = {
      if(nClassCount != 0) {
        // pClassCount.toDouble / nClassCount.toDouble
        0.0
      } else {
        // Double.NaN
        0.0
      }
    }

    nClassCounts.map(x=>getRatio(x))
  })*/

  private val getRatios = udf({(pClassCount: Int, nClassCounts: mutable.WrappedArray[Int]) => Double
    def getRatio(nClassCount: Int): Double = {
      if(nClassCount != 0) {
        pClassCount.toDouble / nClassCount.toDouble
      } else {
        Double.NaN
      }
    }

    nClassCounts.map(x=>getRatio(x))
  })

  //($"neighbors.features", $"ratios"))
  /*private val getValidExamples = udf({(ratios: mutable.WrappedArray[Int], features: mutable.WrappedArray[DenseVector]) =>

    var keepNeighbors = Array[(Double, DenseVector)]()
    for(x <- features.indices) {
      if()

    }

  }) */

  private def generateExamples(label: Double, features: DenseVector, neighborFeatures: mutable.WrappedArray[DenseVector], ratios: mutable.WrappedArray[Double], sampleCount: Int) : Array[Row] ={

    // pick sampleCount examples
    // calculate gap
    // generate example

    def generateExample(index: Int) : Row ={

      // FIXME -- check these
      val gap = if(ratios(index) == Double.NaN) {
        0
      } else if(ratios(index) == 1) {
        Random.nextDouble()
      } else if(ratios(index) > 1) {
        Random.nextDouble() * (1.0 / ratios(index))
      } else {
        (1.0 - ratios(index)) + Random.nextDouble() * ratios(index)
      }

      // val syntheticExample = Vectors.dense(Array(features.toArray, neighborFeatures(index).toArray, gap).tra) // .map(x=>x(0) + x(2) * (x(1)-x(0)))).toDense

      val syntheticExample = Vectors.dense(Array(features.toArray, neighborFeatures(index).toArray).transpose.map(x=>x(0) + (x(1) - x(0) * gap))).toDense

      Row(label, syntheticExample)
    }

    (0 until sampleCount).map(_=>generateExample(Random.nextInt(neighborFeatures.length-1)+1)).toArray
  }


  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val spark = dataset.sparkSession
    import spark.implicits._

    // val majorityDF = dataset.filter(dataset($(labelCol)) =!= minorityClassLabel)

    // println(majorityDF.count, minorityDF.count())

    // FIXME - move to transform function
    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize($(topTreeSize))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel: KNNModel = model.fit(dataset)
    val fullNearestNeighborDF = fitModel.transform(dataset)

    println("*** dfNeighborRatio ****")
    val dfNeighborRatio = fullNearestNeighborDF.withColumn("pClassCount", getSafeNeighborCount($"neighbors.label", $"label")).drop("neighbors")
    dfNeighborRatio.show
    println(dfNeighborRatio.count)


    val model2 = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize($(topTreeSize))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol), "pClassCount"))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel2: KNNModel = model2.fit(dfNeighborRatio)
    val minorityNearestNeighbor = fitModel2.transform(dfNeighborRatio)
    //minorityNearestNeighbor.show()
    //minorityNearestNeighbor.printSchema()
    //println(minorityNearestNeighbor.take(1)(0))

    println("minorityDF")
    val minorityDF = minorityNearestNeighbor.filter(minorityNearestNeighbor($(labelCol)) === minorityClassLabel).filter(minorityNearestNeighbor("pClassCount")=!=0)
    minorityDF.show
    minorityDF.printSchema()
    println(minorityDF.take(1)(0))

    val minorityRatiosDF = minorityDF.withColumn("ratios", getRatios($"pClassCount", $"neighbors.pClassCount")).withColumn("neighborsFeatures", $"neighbors.features")  //getRatios($"pClassCount", $"neighbors.pClassCount"))
    minorityRatiosDF.show()
    minorityRatiosDF.printSchema()

    println("samples to add: " + samplesToAdd)
    println("current counts: " + minorityRatiosDF.count)

    val samplingRate = Math.ceil(samplesToAdd / minorityRatiosDF.count.toDouble).toInt

    println(minorityRatiosDF.select("ratios").take(1)(0))


    //private def generateExamples(label: Double, features: DenseVector, neighborFeatures: mutable.WrappedArray[DenseVector], ratios: mutable.WrappedArray[Double], sampleCount: Int) : Array[Row] ={

    val syntheticExamples: Array[Array[Row]] = minorityRatiosDF.collect().map(x=>generateExamples(x(0).asInstanceOf[Double], x(1).asInstanceOf[DenseVector], x(5).asInstanceOf[mutable.WrappedArray[DenseVector]], x(4).asInstanceOf[mutable.WrappedArray[Double]], samplingRate))

    println("synthetic " + syntheticExamples.length)
    val s2: Array[Row] = syntheticExamples.reduce(_ union _ ) // FIXME use samples to fine tune final size 5%?
    println("s2 count: " + s2.length)

    //spark.createDataFrame(dataset.sparkSession.sparkContext.parallelize(syntheticExamples), dataset.schema)
    spark.createDataFrame(dataset.sparkSession.sparkContext.parallelize(syntheticExamples.reduce(_ union _ )), dataset.schema)

    //val minorityRatiosFiltered = minorityRatiosDF.withColumn("filtered", getValidExamples($"neighbors.features", $"ratios"))
    //minorityRatiosFiltered.show
    //minorityRatiosFiltered.printSchema()


    /*println("nearestNeighborDF")
    val nearestNeighborDF = fitModel.transform(minorityDF)
    nearestNeighborDF.show
    nearestNeighborDF.printSchema()
    println(nearestNeighborDF.take(1)(0))*/

    // dataset.toDF
    /*val df = dataset.toDF
    val spark = df.sparkSession
    import spark.implicits._

    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    // val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString



    val minClassCount = minorityDF.count // counts.take(1)(0)(1).toString.toInt
    val maxClassCount = majorityDF.count // counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    // println(minClassLabel, minClassCount)
    // println(maxClassLabel, maxClassCount)

    val leafSize = 100
    val kValue = 5
    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(kValue + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f: KNNModel = model.fit(df)

    val t = f.transform(minorityDF)//.sort("index")
    println("*** first knn ****")
    t.show


    println("*** dfNeighborRatio ****")
    val dfNeighborRatio = t.withColumn("pClassCount", getSafeNeighborCount($"neighbors.label", $"label"))//.sort("index")//.drop("neighbors")
    dfNeighborRatio.show
    println(dfNeighborRatio.count)

    val dfNeighborRatioFiltered = dfNeighborRatio.filter(dfNeighborRatio("pClassCount")=!=0)
    dfNeighborRatioFiltered.show() ///FIXME - modified algorithm to make it more spark friendly
    println(dfNeighborRatio.count)


    println("minClassCount: " + dfNeighborRatioFiltered.count())
    println("maxClassCount: " + maxClassCount)
    println("to add: " + (maxClassCount.toInt - minClassCount.toInt))
    println("ratio: " + maxClassCount.toInt / dfNeighborRatioFiltered.count().toDouble)
    println("data count: " + dfNeighborRatioFiltered.count())

    val neighborsSampled = dfNeighborRatioFiltered.withColumn("rnd", randUdf())

    println(neighborsSampled.count)
    println("*** dfNeighborSingle ****")
    val dfNeighborSingle = neighborsSampled.withColumn("n", getRandomNeighbor($"neighbors.label", $"neighbors.features", $"rnd")) //.sort("index") //.drop("neighbors")
    dfNeighborSingle.show
    println(dfNeighborSingle.count)

    println("*** dfWithRandomNeighbor ****")
    val dfWithRandomNeighbor = dfNeighborSingle.withColumn("nLabel", $"n._1").withColumn("nFeatures", $"n._2") //.sort("index")//.drop("n").drop("neighbors").sort("index")
    dfWithRandomNeighbor.show
    println(dfWithRandomNeighbor.count)


    println("*** columns renamed ****")
    val dfRenamed = dfWithRandomNeighbor.withColumnRenamed("label", "originalLabel")
      .withColumnRenamed("features", "originalFeatures")
      .withColumnRenamed("nLabel", "label")
      .withColumnRenamed("nFeatures", "features")
      .withColumnRenamed("neighbors", "originalNeighbors")
    dfRenamed.show
    println(dfRenamed.count)
    println("*** second knn ****")
    val t2 = f.transform(dfRenamed).withColumnRenamed("neighbors", "nNeighbors")//.sort("index")
    t2.show
    t2.printSchema()
    println(t2.count)

    val dfNeighborRatio2 = t2.withColumn("nClassCount", getSafeNeighborCount($"nNeighbors.label", $"label"))
      .drop("originalNeighbors").drop("n").drop("nNeighbors").withColumn("featureRnd", randFeatureUdf($"features"))
    dfNeighborRatio2.show
    println(dfNeighborRatio2.count)

    val result = dfNeighborRatio2.withColumn("example", generateExample($"originalLabel", $"label", $"originalFeatures", $"features", $"pClassCount", $"nClassCount", $"featureRnd"))

    println("final: " + result.count)
    result.show

    result // FIXME - check what is really getting returned*/

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
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString.toDouble
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble).take(1)

    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected, x))
      .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, datasetSelected.filter(datasetSelected($(labelCol))===clsList(x._3)).count(), majorityClassCount, $(samplingRatios))))

    val balanecedDF = datasetIndexed.select($(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol)).repartition(1)


  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): SafeLevelSMOTEModel = {
    val copied = new SafeLevelSMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}


/** Estimator Parameters*/
private[ml] trait SafeLevelSMOTEParams extends SafeLevelSMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class SafeLevelSMOTE(override val uid: String) extends Estimator[SafeLevelSMOTEModel] with SafeLevelSMOTEParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): SafeLevelSMOTEModel = {
    val model = new SafeLevelSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): SafeLevelSMOTE = defaultCopy(extra)

}

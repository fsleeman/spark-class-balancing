package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType




/** Transformer Parameters*/
private[ml] trait SafeLevelSMOTEModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class SafeLevelSMOTEModel private[ml](override val uid: String) extends Model[SafeLevelSMOTEModel] with SafeLevelSMOTEModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  private val getSafeNeighborCount = udf((array: mutable.WrappedArray[Int], minorityClassLabel: Int) => {
    def isMajorityNeighbor(x1: Int, x2: Int): Int = {
      if(x1 == x2) {
        1
      } else {
        0
      }
    }
    array.tail.map(x=>isMajorityNeighbor(minorityClassLabel, x)).sum
  })


  private val getRandomNeighbor = udf((labels: mutable.WrappedArray[Int], features: mutable.WrappedArray[DenseVector], rnd: Double) => {
    val randomIndex = (rnd * labels.length-1).toInt + 1
    //Vector(labels(randomIndex), features(randomIndex))
    (labels(randomIndex), features(randomIndex))
  })

  private val randUdf = udf({() => Random.nextDouble()})
  private val randFeatureUdf = udf({(array: DenseVector) =>
    Vectors.dense(Array.fill(array.size)(Random.nextDouble())).toDense
  })

  private val generateExample = udf((pLabel: Int, nLabel: Int, pFeatures: DenseVector, nFeatures:DenseVector, pLabelCount: Int, nLabelCount: Int, rnds2: DenseVector) => {
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

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF
    val spark = df.sparkSession
    import spark.implicits._

    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    // val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString

    val minorityDF = df.filter(df("label") === minClassLabel)
    val majorityDF = df.filter(df("label") =!= minClassLabel)

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

    val t = f.transform(minorityDF).sort("index")
    println("*** first knn ****")
    t.show


    println("*** dfNeighborRatio ****")
    val dfNeighborRatio = t.withColumn("pClassCount", getSafeNeighborCount($"neighbors.label", $"label")).sort("index")//.drop("neighbors")
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
    val dfNeighborSingle = neighborsSampled.withColumn("n", getRandomNeighbor($"neighbors.label", $"neighbors.features", $"rnd")).sort("index") //.drop("neighbors")
    dfNeighborSingle.show
    println(dfNeighborSingle.count)

    println("*** dfWithRandomNeighbor ****")
    val dfWithRandomNeighbor = dfNeighborSingle.withColumn("nLabel", $"n._1").withColumn("nFeatures", $"n._2")//.sort("index")//.drop("n").drop("neighbors").sort("index")
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
    val t2 = f.transform(dfRenamed).withColumnRenamed("neighbors", "nNeighbors").sort("index")
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

    df
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

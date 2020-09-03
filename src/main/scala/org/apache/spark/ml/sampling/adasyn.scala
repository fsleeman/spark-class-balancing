package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.collection.mutable
import scala.util.Random







/** Transformer Parameters*/
private[ml] trait ADASYNModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class ADASYNModel private[ml](override val uid: String) extends Model[ADASYNModel] with ADASYNModelParams {
  def this() = this(Identifiable.randomUID("classBalancer"))

  def generateExamples(row: Row): Array[Array[Double]] = {
    val label = row(1).toString.toInt
    val examplesToCreate = row(5).asInstanceOf[Long].toInt
    val neighborLabels = row(6).asInstanceOf[mutable.WrappedArray[Int]]
    val neighborFeatures: mutable.Seq[DenseVector] = row(7).asInstanceOf[mutable.WrappedArray[DenseVector]]

    if (neighborLabels.tail.contains(label)) {
      // skip self instance
      var minorityIndicies = Array[Int]()
      for (x <- 1 until neighborLabels.length) {
        if (neighborLabels(x) == label) {
          minorityIndicies = minorityIndicies :+ x
        }
      }

      val randomIndicies = (0 until examplesToCreate).map(_ => minorityIndicies.toVector(Random.nextInt(minorityIndicies.length)))
      (0 until examplesToCreate).map(x => neighborFeatures(randomIndicies(x)).toArray).toArray
    } else {
      val features: Array[Double] = neighborFeatures.head.toArray
      (0 until examplesToCreate).map(x => features).toArray
    }
  }


  def oversampleClass(dataset: Dataset[_], minorityClassLabel: String, samplesToAdd: Int): DataFrame = {
    val df = dataset.toDF
    import df.sparkSession.implicits._

    val minorityDF = df.filter(df("label") === minorityClassLabel)
    val majorityDF = df.filter(df("label") =!= minorityClassLabel)
    val minorityClassCount = minorityDF.count
    val majorityClassCount = majorityDF.count

    val threshold = 1.0
    val beta = 1.0 // final balance level, might need to adjust from the original paper

    val imbalanceRatio = minorityClassLabel.toDouble / majorityClassCount.toDouble

    //val G = (majorityClassCount - minorityClassCount) * beta
    val G = samplesToAdd

    val leafSize = 100
    val kValue = 5
    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8) /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(kValue + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f: KNNModel = model.fit(df).setDistanceCol("distances")
    val t = f.transform(minorityDF)
    t.show

    val getMajorityNeighborRatio = udf((array: mutable.WrappedArray[Int]) => {
      def isMajorityNeighbor(x1: Int, x2: Int): Int = {
        if (x1 == x2) {
          0
        } else {
          1
        }
      }
      array.tail.map(x => isMajorityNeighbor(array.head, x)).sum / kValue.toDouble
    })


    val collected = t.select($"neighbors.label")
    collected.show
    collected.printSchema()

    val dfNeighborRatio = t.withColumn("neighborClassRatio", getMajorityNeighborRatio($"neighbors.label")).drop("distances") //.drop("neighbors")
    dfNeighborRatio.show

    val neighborRatioSum = dfNeighborRatio.agg(sum("neighborClassRatio")).first.get(0).toString.toDouble
    println(neighborRatioSum)

    val getSampleCount = udf((density: Double) => {
      Math.round(density / neighborRatioSum * G.toDouble)
    })

    val adjustedRatios = dfNeighborRatio.withColumn("samplesToAdd", getSampleCount($"neighborClassRatio")).withColumn("labels", $"neighbors.label").withColumn("neighborFeatures", $"neighbors.features")
    adjustedRatios.show
    adjustedRatios.printSchema()

    val samplesToAddSum = adjustedRatios.agg(sum("samplesToAdd")).first.get(0).toString.toDouble
    println("majority count: " + majorityClassCount)
    println("minority count: " + minorityClassCount)
    println("samples to add: " + samplesToAddSum) // FIXME - double check the size, wont be exactly the right number because of rounding

    adjustedRatios.withColumn("labels", $"neighbors.label").withColumn("neighborFeatures", $"neighbors.features").show

    val syntheticExamples: Array[Array[Array[Double]]] = adjustedRatios.collect.map(x => generateExamples(x))

    println(syntheticExamples.length)
    val totalExamples = syntheticExamples.flatMap(x => x.toSeq).map(x => (0, minorityClassLabel.toInt, Vectors.dense(x).toDense))


    println("total: " + totalExamples.length)

    val bar = df.sparkSession.createDataFrame(df.sparkSession.sparkContext.parallelize(totalExamples))
    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    // df.union(bar2)
    bar2

  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF()
    import df.sparkSession.implicits._

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

    // println(minClassLabel, minClassCount)
    // println(maxClassLabel, maxClassCount)

  }



  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): ADASYNModel = {
    val copied = new ADASYNModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}

/** Estimator Parameters*/
private[ml] trait ADASYNParams extends ADASYNModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class ADASYN(override val uid: String) extends Estimator[ADASYNModel] with ADASYNParams {
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): ADASYNModel = {
    val model = new ADASYNModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ADASYN = defaultCopy(extra)

}

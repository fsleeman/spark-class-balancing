package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.sampling.utilities._
import org.apache.spark.ml.sampling.utils.getCountsByClass
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.collection.mutable
import scala.util.Random



/** Transformer Parameters*/
private[ml] trait ADASYNModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {

}

/** Transformer */
class ADASYNModel private[ml](override val uid: String) extends Model[ADASYNModel] with ADASYNModelParams {
  def this() = this(Identifiable.randomUID("adasyn"))

  def generateExamples(row: Row): Array[Row] = {
    val label = row(0).toString.toDouble
    val examplesToCreate = row(4).asInstanceOf[Long].toInt
    val neighborLabels = row(5).asInstanceOf[mutable.WrappedArray[Double]]
    val neighborFeatures: mutable.Seq[DenseVector] = row(6).asInstanceOf[mutable.WrappedArray[DenseVector]]

    if (neighborLabels.tail.contains(label)) {
      // skip self instance
      var minorityIndicies = Array[Int]()
      for (x <- 1 until neighborLabels.length) {
        if (neighborLabels(x) == label) {
          minorityIndicies = minorityIndicies :+ x
        }
      }

      val randomIndicies = (0 until examplesToCreate).map(_ => minorityIndicies.toVector(Random.nextInt(minorityIndicies.length)))
      // (0 until examplesToCreate).map(x => Row(label, neighborFeatures(randomIndicies(x)))).toArray
      print("** neighbor length: " + neighborFeatures.length + " ")
      for(x<-randomIndicies){
        print(x + " ")
      }
      println("**")
      //randomIndicies.map(x => Row(label, createSmoteStyleExample(neighborFeatures.head, neighborFeatures(randomIndicies(x))))).toArray
      randomIndicies.map(x => Row(label, createSmoteStyleExample(neighborFeatures.head, neighborFeatures(x)))).toArray
    } else {
      // just in case we end up here
      val features: Array[Double] = neighborFeatures.head.toArray
      (0 until examplesToCreate).map(_ => Row(label, Vectors.dense(features).toDense)).toArray
    }
  }

  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame = {
    val spark = dataset.sparkSession
    import dataset.sparkSession.implicits._

    val minorityDF = dataset.filter(dataset($(labelCol)) === minorityClassLabel)
    val G = samplesToAdd

    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), dataset.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel: KNNModel = model.fit(dataset).setDistanceCol("distances")
    val minorityDataNeighbors = fitModel.transform(minorityDF)
    println("*** neighbors")
    minorityDataNeighbors.show()

    val getMajorityNeighborRatio = udf((array: mutable.WrappedArray[String]) => {
      def isMajorityNeighbor(x1: String, x2: String): Int = {
        if (x1 == x2) {
          0
        } else {
          1
        }
      }
      array.tail.map(x => isMajorityNeighbor(array.head, x)).sum / $(k).toFloat
    })

    val dfNeighborRatio = minorityDataNeighbors.withColumn("neighborClassRatio", getMajorityNeighborRatio($"neighbors.label")).drop("distances")
    val neighborRatioSum = dfNeighborRatio.agg(sum("neighborClassRatio")).first.get(0).toString.toDouble

    val getSampleCount = udf((density: Double) => {
      Math.round(density / neighborRatioSum * G.toDouble)
    })

    val adjustedRatios = dfNeighborRatio.withColumn("samplesToAdd", getSampleCount($"neighborClassRatio")).withColumn("labels", $"neighbors.label").withColumn("neighborFeatures", $"neighbors.features")
    println("*** adjusted")
    adjustedRatios.show()
    val syntheticExamples: Array[Array[Row]] = adjustedRatios.collect.map(x => generateExamples(x))
    val totalExamples: Array[Row] = syntheticExamples.flatMap(x => x.toSeq)

    println("~~ added for " + minorityClassLabel + ": " + totalExamples.length)

    spark.createDataFrame(dataset.sparkSession.sparkContext.parallelize(totalExamples), dataset.schema)
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

    //val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected.filter(datasetSelected($(labelCol))===clsList(x))))
    //  .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, x._2.count, majorityClassCount, $(samplingRatios))))

    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected))
      .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, x._2.filter(x._2($(labelCol))===x._1.toDouble).count(), majorityClassCount, $(samplingRatios))))

    val balanecedDF = datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))//.repartition(1)
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
  def this() = this(Identifiable.randomUID("adasyn"))

  override def fit(dataset: Dataset[_]): ADASYNModel = {
    val model = new ADASYNModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): ADASYN = defaultCopy(extra)

}

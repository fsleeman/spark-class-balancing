package org.apache.spark.ml.sampling

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasSeed}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.sampling.utilities.{ClassBalancingRatios, HasLabelCol, UsingKNN, calculateToTreeSize, getSamplesToAdd}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.mutable
import scala.util.Random
import org.apache.spark.ml.sampling.utils._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType

/** Transformer Parameters*/
private[ml] trait BorderlineSMOTEModelParams extends Params with HasFeaturesCol with HasLabelCol with UsingKNN with ClassBalancingRatios {
  /**
    * Param for method
    * @group param
    */
  final val method: Param[Int] = new Param[Int](this, "mode", "Borderline SMOTE method, 1 - multi-class, 2 - binary only")

  /** @group getParam */
  final def setMethod(value: Int): this.type = set(method, value)

  setDefault(method -> 1)
}

/** Transformer */
class BorderlineSMOTEModel private[ml](override val uid: String) extends Model[BorderlineSMOTEModel] with BorderlineSMOTEModelParams {
  def this() = this(Identifiable.randomUID("borderlineSmote"))

  private val isDanger: UserDefinedFunction = udf(f = (neighbors: mutable.WrappedArray[Double]) => {
    val nearestClasses = neighbors

    val currentClass = nearestClasses(0)
    val majorityNeighbors = nearestClasses.tail.map(x => if (x == currentClass) 0 else 1).sum
    val numberOfNeighbors = nearestClasses.length - 1
    if (numberOfNeighbors / 2 <= majorityNeighbors && majorityNeighbors < numberOfNeighbors) {
      true
    } else {
      false
    }
  })

  def getNewSample(current: DenseVector, i: DenseVector, range: Double) : DenseVector = {
    val xx: Array[Array[Double]] = Array(current.values, i.values)
    val sample: Array[Double] = xx.transpose.map(x=>x(0) + Random.nextDouble * range * (x(1) + x(0)))
    Vectors.dense(sample).toDense
  }

  def generateSamples(neighbors: Seq[DenseVector], s: Int, range: Double, label: Double): Array[Row] = {
    val current = neighbors.head
    val selected: Seq[DenseVector] = Random.shuffle(neighbors.tail).take(s)
    val rows = selected.map(x=>getNewSample(current, x, range)).map(x=>Row(label, x))
    rows.toArray
  }

  def oversample(dataset: Dataset[_], minorityClassLabel: Double, samplesToAdd: Int): DataFrame ={
    val spark = dataset.sparkSession
    import dataset.sparkSession.implicits._

    // step 1
    //  For each minority example, calculate the m nn's in training set
    val minorityDF = dataset.filter(dataset($(labelCol))===minorityClassLabel)

    val model = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), dataset.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel = model.fit(dataset)
    val nearestNeighborDF = fitModel.transform(minorityDF)

    // step 2
    // Find DANGER examples: if m'=m (noise), m/2 < m' < m (DANGER), 0 <= m' <= m/2 (safe)

    val dfDanger = nearestNeighborDF.filter(isDanger(nearestNeighborDF("neighbors.label")))
    val s = if(dfDanger.count == 0) {
      0
    } else {
      (samplesToAdd / dfDanger.count.toDouble).ceil.toInt
    }

    // step 3
    // For all DANGER examples, find k nearest examples from minority class
    val model2 = new KNN().setFeaturesCol($(featuresCol))
      .setTopTreeSize(calculateToTreeSize($(topTreeSize), minorityDF.count()))
      .setTopTreeLeafSize($(topTreeLeafSize))
      .setSubTreeLeafSize($(subTreeLeafSize))
      .setK($(k) + 1) // include self example
      .setAuxCols(Array($(labelCol), $(featuresCol)))
      .setBalanceThreshold($(balanceThreshold))

    val fitModel2 = model2.fit(minorityDF)
    val minorityNearestNeighborDF = fitModel2.transform(dfDanger.drop("neighbors"))

    // step 4

    /*** generate (s * DANGER examples)
      *  p'i = borderline minority examples
      *  s = integer between 1 and k
      *  for each p'i, randomly select its s nearest neighbors in P and find distances to p'i.
      *  Get rand (0,1) - r for each so s synthetic examples are created using p'i * r(j) * difference(j)
      *
      ***/
    val minorityNeighborFeatures: DataFrame = minorityNearestNeighborDF.select($"neighbors.features")
    val result: Array[Array[Row]] = minorityNeighborFeatures.collect
      .map(row=>generateSamples(row.getValuesMap[Any](row.schema.fieldNames)($(featuresCol))
        .asInstanceOf[mutable.WrappedArray[DenseVector]], s, 1.0, minorityClassLabel))

    // mode = 1: borderlineSMOTE1: use minority NNs
    // mode = 2: borderlineSMOTE2: use minority NNs AND majority NNs
    val oversamplingRows: Array[Row] = if($(method) == 1) {
      val r = scala.util.Random
      r.shuffle(result.flatMap(x => x.toSeq).toList).take(samplesToAdd).toArray
    }
    else {
      val negativeDF = dataset.filter(dataset($(labelCol))=!=minorityClassLabel)

      val modelNegative = new KNN().setFeaturesCol($(featuresCol))
        .setTopTreeSize(calculateToTreeSize($(topTreeSize), negativeDF.count()))
        .setTopTreeLeafSize($(topTreeLeafSize))
        .setSubTreeLeafSize($(subTreeLeafSize))
        .setK($(k) + 1) // include self example
        .setAuxCols(Array($(labelCol), $(featuresCol)))
        .setBalanceThreshold($(balanceThreshold))

      val fNegative = modelNegative.fit(negativeDF)
      val tNegative = fNegative.transform(dfDanger.drop("neighbors"))

      val nearestNegativeExamples: DataFrame = tNegative.select($"neighbors.features")
      val nearestNegativeSamples: Array[Array[Row]] = nearestNegativeExamples.collect
        .map(row=>generateSamples(row.getValuesMap[Any](row.schema.fieldNames)($(featuresCol))
          .asInstanceOf[mutable.WrappedArray[DenseVector]], s, 0.5, minorityClassLabel))
      val nearestNegativeRows: Array[Row] = nearestNegativeSamples.flatMap(x=>x.toSeq)

      result.flatMap(x => x.toSeq) ++ nearestNegativeRows
    }

    spark.createDataFrame(spark.sparkContext.parallelize(oversamplingRows), dataset.schema)
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

    val datasetSelected = datasetIndexed.select($(labelCol), $(featuresCol))
    val counts = getCountsByClass(datasetSelected.sparkSession, $(labelCol), datasetSelected.toDF).sort("_2")
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString.toDouble
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val clsList: Array[Double] = counts.select("_1").filter(counts("_1") =!= majorityClassLabel).collect().map(x=>x(0).toString.toDouble)

    val clsDFs = clsList.indices.map(x=>(clsList(x), datasetSelected, x))
      .map(x=>oversample(x._2, x._1, getSamplesToAdd(x._1.toDouble, datasetSelected.filter(datasetSelected($(labelCol))===clsList(x._3)).count(), majorityClassCount, $(samplingRatios))))

    val balanecedDF = datasetIndexed.select( $(labelCol), $(featuresCol)).union(clsDFs.reduce(_ union _))
    val restoreLabel = udf((label: Double) => labelMapReversed(label))

    balanecedDF.withColumn("originalLabel", restoreLabel(balanecedDF.col($(labelCol)))).drop( $(labelCol))
      .withColumnRenamed("originalLabel",  $(labelCol))//.repartition(1)
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def copy(extra: ParamMap): BorderlineSMOTEModel = {
    val copied = new BorderlineSMOTEModel(uid)
    copyValues(copied, extra).setParent(parent)
  }

}

/** Estimator Parameters*/
private[ml] trait BorderlineSMOTEParams extends BorderlineSMOTEModelParams with HasSeed {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    schema
  }
}

/** Estimator */
class BorderlineSMOTE(override val uid: String) extends Estimator[BorderlineSMOTEModel] with BorderlineSMOTEParams {
  def this() = this(Identifiable.randomUID("borderlineSmote"))

  override def fit(dataset: Dataset[_]): BorderlineSMOTEModel = {
    val model = new BorderlineSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): BorderlineSMOTE = defaultCopy(extra)

}

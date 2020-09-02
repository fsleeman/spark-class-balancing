package org.apache.spark.ml.sampling

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.knn.{KNN, KNNModel}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasSeed}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random
import org.apache.spark.ml.sampling.utils._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.StructType




/** Transformer Parameters*/
private[ml] trait BorderlineSMOTEModelParams extends Params with HasFeaturesCol with HasInputCols {

}

/** Transformer */
class BorderlineSMOTEModel private[ml](override val uid: String) extends Model[BorderlineSMOTEModel] with BorderlineSMOTEModelParams {
  def this() = this(Identifiable.randomUID("borderlineSmote"))

  /*************************************************/
  /************borderline SMOTE*****************/
  /*************************************************/
  val isDanger: UserDefinedFunction = udf((neighbors: mutable.WrappedArray[Element]) => {
    val nearestClasses = neighbors.asInstanceOf[mutable.WrappedArray[Int]]
    val currentClass = nearestClasses(0)
    val majorityNeighbors = nearestClasses.tail.map(x=>if(x==currentClass) 0 else 1).sum
    val numberOfNeighbors = nearestClasses.length - 1

    if(numberOfNeighbors / 2 <= majorityNeighbors && majorityNeighbors < numberOfNeighbors) {
      true
    } else {
      false
    }
  })


  /*def borderlineSmote1(spark: SparkSession, dfIn: DataFrame): DataFrame = {
    borderlineSmote(spark, dfIn, 1)
  }

  def borderlineSmote2(spark: SparkSession, dfIn: DataFrame): DataFrame = {
    borderlineSmote(spark, dfIn, 2)
  }*/
  
  val getLabel = udf((neighbors: Seq[Row]) => scala.util.Try(
    neighbors.map(_.getAs[Int]("label"))
  ).toOption)


  def getNewSample(current: DenseVector, i: DenseVector, range: Double) : DenseVector = {
    val xx: Array[Array[Double]] = Array(current.values, i.values)
    val sample: Array[Double] = xx.transpose.map(x=>x(0) + Random.nextDouble * range * (x(1)+x(0)))
    Vectors.dense(sample).toDense
  }

  def generateSamples(neighbors: Seq[DenseVector], s: Int, range: Double, label: Int): Array[Row] = {

    val current = neighbors.head
    val selected: Seq[DenseVector] = Random.shuffle(neighbors.tail).take(s)
    val rows = selected.map(x=>getNewSample(current, x, range)).map(x=>Row(0, label, x))
    rows.toArray
  }

   def transform2(dataset: Dataset[_]): DataFrame = {
    val mode = 1 // FIXME - add parameter

    val df = dataset.toDF // .filter((dataset("label") === 1) || (dataset("label") === 5)).toDF // FIXME
    import df.sparkSession.implicits._

    val m = 5   // k-value
    val leafSize = 1000

    val counts = getCountsByClass(df.sparkSession, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt


    val clsList: Array[Int] = counts.select("_1").filter(counts("_1") =!= maxClassLabel).collect().map(x=>x(0).toString.toInt)
    // val clsDFs = clsList.indices.map(x=>oversample(df.filter(df("label")===clsList(x)), maxClassCount))

    println("Majority class: " + maxClassLabel)
    for(x<-clsList) {
      println("cls: " + x)
    }

    val samplesToAdd = maxClassCount - minClassCount

    dataset.toDF

  }

  def oversampleClass(dataset: Dataset[_], minorityClassLabel: String, samplesToAdd: Int): DataFrame ={
    val df = dataset.toDF //filter((dataset("label") === minorityClassLabel) || (dataset("label") === 4) || (dataset("label") === 2) || (dataset("label") === 3) || (dataset("label") === 5) || (dataset("label") === 6) || (dataset("label") === 7)).toDF // FIXME
    import df.sparkSession.implicits._
    val m = 5   // k-value
    val leafSize = 1000

    //val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)
    // println("minority count: " + minClassCount)


    // step 1
    //  For each minority example, calculate the m nn's in training set
    val minorityDF = df.filter(df("label")===minorityClassLabel)

    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(m + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f = model.fit(df)
    val t = f.transform(minorityDF)


    // step 2
    // Find DANGER examples: if m'=m (noise), m/2 < m' < m (DANGER), 0 <= m' <= m/2 (safe)
    t.show()
    t.printSchema()


    println("Is danger")
    val dfDanger = t.filter(isDanger(t("neighbors").getItem("label")))
    dfDanger.show()
    println("minorityDF:" + minorityDF.count())
    println("danger count:" + dfDanger.count)

    println("s value: " + samplesToAdd / dfDanger.count.toDouble)
    val s = (samplesToAdd / dfDanger.count.toDouble).ceil.toInt
    println(s)


    // step 3
    // For all DANGER examples, find k nearest examples from minority class
    val kValue = 5 /// FIXME - check the k/m values
    val model2 = new KNN().setFeaturesCol("features")
      // .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeSize(10)   /// FIXME - check this size for small datasets
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(kValue + 1) // include self example
      .setAuxCols(Array("label", "features"))
    // val f2 = model2.fit(df.filter(df("label")===minorityClassLabel))
   val f2 = model2.fit(minorityDF)
    val t2 = f2.transform(dfDanger.drop("neighbors"))

    // step 4

    /*** generate (s * DANGER examples)
      *  p'i = borderline minority examples
      *  s = integer between 1 and k
      *  for each p'i, randomly select its s nearest neighbors in P and find distances to p'i.
      *  Get rand (0,1) - r for each so s synthetic examples are created using p'i * r(j) * difference(j)
      *
      ***/
    val xxxx: DataFrame = t2.select($"neighbors.features")
    val result: Array[Array[Row]] = xxxx.collect.map(row=>generateSamples(row.getValuesMap[Any](row.schema.fieldNames)("features").asInstanceOf[mutable.WrappedArray[DenseVector]], s, 1.0, minorityClassLabel.toInt))

    // mode = 1: borderlineSMOTE1: use minority NNs
    // mode = 2: borderlineSMOTE2: use minority NNs AND majority NNs
    val mode = 1 // FIXME - add parameter
    val fooX: Array[Row] = if(mode == 1) { // FIXME - dont use mode 2, only works directly with binary case
      val r = scala.util.Random
      r.shuffle(result.flatMap(x => x.toSeq).toList).take(samplesToAdd).toArray
    }
    else {
      val modelNegative = new KNN().setFeaturesCol("features") // FIXME - check if the right DFs are used in NNs
        .setTopTreeSize(df.count().toInt / 8)   /// FIXME - parameter
        .setTopTreeLeafSize(leafSize)
        .setSubTreeLeafSize(leafSize)
        .setK(kValue + 1) // include self example
        .setAuxCols(Array("label", "features"))

      val fNegative = modelNegative.fit(df.filter(df("label")=!=minorityClassLabel))
      val tNegative = fNegative.transform(dfDanger.drop("neighbors"))

      val nearestNegativeExamples: DataFrame = tNegative.select($"neighbors.features")
      val nearestNegativeSamples: Array[Array[Row]] = nearestNegativeExamples.collect.map(row=>generateSamples(row.getValuesMap[Any](row.schema.fieldNames)("features").asInstanceOf[mutable.WrappedArray[DenseVector]], s, 0.5, minorityClassLabel.toInt))
      val nearestNegativeRows: Array[Row] = nearestNegativeSamples.flatMap(x=>x.toSeq)

      result.flatMap(x => x.toSeq) ++ nearestNegativeRows
    }

    println("len: " + xxxx.count())
    println("foo: " + fooX.length)

    val foo: Array[(Long, Int, DenseVector)] = fooX.map(x=>x.toSeq).map(x=>(x.head.toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector]))
    val bar = df.sparkSession.createDataFrame(df.sparkSession.sparkContext.parallelize(foo))
    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    //val all = df.union(bar2)
    //all.show()
    //println("all: " + all.count())

    //all
    bar2
  }

  override def transform(dataset: Dataset[_]): DataFrame = {



    val counts = getCountsByClass(dataset.sparkSession, "label", dataset.toDF).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val majorityClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val majorityClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val minorityClasses = counts.collect.map(x=>(x(0).toString, x(1).toString.toInt)).filter(x=>x._1!=majorityClassLabel)
    for(x<-minorityClasses) {
      println("minority label: " + x)
    }

    val results: DataFrame = minorityClasses.map(x=>oversampleClass(dataset, x._1, majorityClassCount - x._2)).reduce(_ union _).union(dataset.toDF())

//    val results: DataFrame = oversampleClass(dataset, 1.toString, majorityClassCount)

    // val samplesToAdd = maxClassCount - minClassCount
    // println("samples to add: " + samplesToAdd)
    // val results: DataFrame = oversampleClass(dataset, 1.toString, samplesToAdd)

    println("dataset: " + dataset.count)
    println("added: " + results.count)
    // println("result: " + result.count)

    // val clsList: Array[Int] = counts.select("_1").filter(counts("_1") =!= maxClassLabel).collect().map(x=>x(0).toString.toInt)
    // val clsDFs = clsList.indices.map(x=>oversample(df.filter(df("label")===clsList(x)), maxClassCount))
    
    //
    results

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
  def this() = this(Identifiable.randomUID("sampling"))

  override def fit(dataset: Dataset[_]): BorderlineSMOTEModel = {
    val model = new BorderlineSMOTEModel(uid).setParent(this)
    copyValues(model)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): BorderlineSMOTE = defaultCopy(extra)

}

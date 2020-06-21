package org.apache.spark.ml.sampling

import org.apache.spark.ml.knn.KNN
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{desc, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable
import scala.util.Random

import org.apache.spark.ml.sampling.utils._


class BorderlineSmote {

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


  def borderlineSmote1(spark: SparkSession, dfIn: DataFrame): DataFrame = {
    borderlineSmote(spark, dfIn, 1)
  }

  def borderlineSmote2(spark: SparkSession, dfIn: DataFrame): DataFrame = {
    borderlineSmote(spark, dfIn, 2)
  }

  def borderlineSmote(spark: SparkSession, dfIn: DataFrame, mode: Int): DataFrame = {
    import spark.implicits._

    val df = dfIn.filter((dfIn("label") === 1) || (dfIn("label") === 5)) // FIXME
    val m = 5   // k-value
    val leafSize = 1000

    val counts = getCountsByClass(spark, "label", df).sort("_2")
    val minClassLabel = counts.take(1)(0)(0).toString
    val minClassCount = counts.take(1)(0)(1).toString.toInt
    val maxClassLabel = counts.orderBy(desc("_2")).take(1)(0)(0).toString
    val maxClassCount = counts.orderBy(desc("_2")).take(1)(0)(1).toString.toInt

    val samplesToAdd = maxClassCount - minClassCount
    println("Samples to add: " + samplesToAdd)

    // step 1
    /*** For each minority example, calculate the m nn's in training set***/
    val minorityDF = df.filter(df("label")===minClassLabel)
    val model = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(m + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f = model.fit(df)
    val t = f.transform(minorityDF) // FIXME - check if the nearest neighbor should be ignored or not


    // step 2
    /*** Find DANGER examples: if m'=m (noise), m/2 < m' < m (DANGER), 0 <= m' <= m/2 (safe)***/
    t.show()
    t.printSchema()


    println("Is danger")
    val dfDanger = t.filter(isDanger(t("neighbors").getItem("label")))
    dfDanger.show()
    println(minorityDF.count())
    println(dfDanger.count)

    val s = 3

    // step 3
    /*** For all DANGER examples, find k nearest examples from minority class***/
    val kValue = 5 /// FIXME - check the k/m values
    val model2 = new KNN().setFeaturesCol("features")
      .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
      .setTopTreeLeafSize(leafSize)
      .setSubTreeLeafSize(leafSize)
      .setK(kValue + 1) // include self example
      .setAuxCols(Array("label", "features"))

    val f2 = model2.fit(df.filter(df("label")===minClassLabel))
    val t2 = f2.transform(dfDanger.drop("neighbors"))





    // step 4

    /*** generate (s * DANGER examples)
      *  p'i = borderline minority examples
      *  s = integer between 1 and k
      *  for each p'i, randomly select its s nearest neighbors in P and find distances to p'i.
      *  Get rand (0,1) - r for each so s synthetic examples are created using p'i * r(j) * difference(j)
      *
      ***/

    //val temp: Row = t2.take(1)(0)
    // println(temp)

    //import org.apache.spark.sql._

    //val zz: Int = spark.sparkContext.parallelize(Array(1,2,3)).map(Row(_)).collect()(0).getInt(0)
    //println("zz " + zz)

    val xxxx: DataFrame = t2.select($"neighbors.features")
    //xxxx.show()

    //xxxx.printSchema()

    val result: Array[Array[Row]] = xxxx.collect.map(row=>generateSamples(row.getValuesMap[Any](row.schema.fieldNames)("features").asInstanceOf[mutable.WrappedArray[DenseVector]], s, 1.0, minClassLabel.toInt))

    // mode = 1: borderlineSMOTE1: use minority NNs
    // mode = 2: borderlineSMOTE2: use minority NNs AND majority NNs
    val fooX: Array[Row] = if(mode == 1) {
      result.flatMap(x => x.toSeq)
    }
    else {
      val modelNegative = new KNN().setFeaturesCol("features")
        .setTopTreeSize(df.count().toInt / 8)   /// FIXME - check?
        .setTopTreeLeafSize(leafSize)
        .setSubTreeLeafSize(leafSize)
        .setK(kValue + 1) // include self example
        .setAuxCols(Array("label", "features"))

      val fNegative = modelNegative.fit(df.filter(df("label")=!=minClassLabel))
      val tNegative = fNegative.transform(dfDanger.drop("neighbors"))

      val nearestNegativeExamples: DataFrame = tNegative.select($"neighbors.features")
      val nearestNegativeSamples: Array[Array[Row]] = nearestNegativeExamples.collect.map(row=>generateSamples(row.getValuesMap[Any](row.schema.fieldNames)("features").asInstanceOf[mutable.WrappedArray[DenseVector]], s, 0.5, minClassLabel.toInt))
      val nearestNegativeRows: Array[Row] = nearestNegativeSamples.flatMap(x=>x.toSeq)

      result.flatMap(x => x.toSeq) ++ nearestNegativeRows
    }


    println("len: " + xxxx.count())
    println("foo: " + fooX.length)

    val foo: Array[(Long, Int, DenseVector)] = fooX.map(x=>x.toSeq).map(x=>(x.head.toString.toLong, x(1).toString.toInt, x(2).asInstanceOf[DenseVector]))
    val bar = spark.createDataFrame(spark.sparkContext.parallelize(foo))
    val bar2 = bar.withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    //bar2.show
    //bar2.printSchema()
    //println(bar2.count)

    val all = df.union(bar2)
    //println("***********")
    all.show()
    println("all: " + all.count())
    ////val ooo: DataFrame = getCountsByClass(spark, "label", all)
    //ooo.show
    //println("***********")

    all
  }

  val getLabel = udf((neighbors: Seq[Row]) => scala.util.Try(
    neighbors.map(_.getAs[Int]("label"))
  ).toOption)


  def getNewSample(current: DenseVector, i: DenseVector, range: Double) : DenseVector = {
    val xx: Array[Array[Double]] = Array(current.values, i.values)
    val sample: Array[Double] = xx.transpose.map(x=>x(0) + Random.nextDouble * range * (x(1)-x(0)))
    Vectors.dense(sample).toDense
  }

  def generateSamples(neighbors: Seq[DenseVector], s: Int, range: Double, label: Int): Array[Row] = {

    val current = neighbors.head
    val selected: Seq[DenseVector] = Random.shuffle(neighbors.tail).take(s)
    val rows = selected.map(x=>getNewSample(current, x, range)).map(x=>Row(0, label, x))
    rows.toArray
  }


}

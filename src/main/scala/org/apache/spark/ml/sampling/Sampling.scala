package org.apache.spark.ml.sampling

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{RandomForestClassifier, LinearSVC, OneVsRest}
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.linalg.SparseVector

import scala.io.Source
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.knn.KNN
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.sampling.utilities.{convertFeaturesToVector, getCountsByClass}

import scala.collection.mutable
//FIXME - turn classes back to Ints instead of Doubles
object Sampling {

  var resultIndex = 0
  var results = ""
  var resultArray: Array[Array[String]] = Array()

  // FIXME - define this in another file?
  type NearestClassResult = (Int, Array[Int]) //class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) //index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)

  /*def maxValue(a: Double, b:Double): Double ={
    if(a >= b) { a }
    else { b }
  }
  def minValue(a: Double, b:Double): Double ={
    if(a <= b) { a }
    else { b }
  }*/

  def printArray(a: Array[Double]) ={
    for(x<-a) {
      print(x + " ")
    }
    println("")
  }


  def calculateClassifierResults(distinctClasses: DataFrame, confusionMatrix: DataFrame, labels: Array[Double]): Array[String]={//String ={

    println("LABELS")
    for(x<-labels) {
      println(x)
    }
    distinctClasses.printSchema()
    distinctClasses.show(100)

    //FIXME - don't calculate twice
    // val classLabels = distinctClasses.collect().map(x => x.toSeq.last.toString.toDouble)

    //val maxLabel: Double = classLabels.max
    //val minLabel: Double = classLabels.min
    val numberOfClasses = labels.length.toDouble
    //val classCount = confusionMatrix.columns.length - 1
    //val testLabels = distinctClasses.map(_.getAs[String]("label")).map(x => x.toDouble).collect().sorted
    println("labels: " + labels.length + " numberOfClasses: " + numberOfClasses)

    val rows: Array[Array[Double]] = confusionMatrix.collect.map(_.toSeq.toArray.map(_.toString.toDouble))

    // val labels = Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    val classMaps = rows.indices.map(x=>(rows(x).head, x)).toMap
    println(classMaps)
    for(x<-classMaps) {
  //    println(x._1, x._2)
    }

    var updatedRows = Array[Array[Double]]()
    for(x<-labels.indices) {
      if(classMaps.contains(labels(x))) {
        // println(rows(classMaps(x)).toString)
    //    printArray(rows(classMaps(x)))
        updatedRows = updatedRows :+ rows(classMaps(x))

      } else {
        //updatedRows +: Array[Double](labels(x)) ++ labels.indices.map(_=>0.0)
        val emptyRow = Array[Double](labels(x)) ++ labels.indices.map(_=>0.0) //, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) // Array[Double](labels(x)) ++ labels.indices.map(_=>0.0)
        // println(emptyRow.toString)
      //  printArray(emptyRow)
        updatedRows = updatedRows :+ emptyRow

      }
    }

    println("updated rows")
    for(x<-updatedRows) {
      printArray(x)
    }

    println("****")

    val totalCount = updatedRows.map(x => x.tail.sum).sum
    // val classMaps: Array[(Int, Double)] = testLabels.zipWithIndex.map(x => (x._2, x._1))

    /*println("classMaps")
    for(x<-classMaps) {
      println(x._1, x._2)
    }*/

    var AvAvg = 0.0
    var MAvG = 1.0
    var RecM = 0.0
    var PrecM = 0.0
    var Precu = 0.0
    var Recu = 0.0
    var FbM = 0.0
    var Fbu = 0.0
    var AvFb = 0.0
    var CBA = 0.0

    var tSum = 0.0
    var pSum = 0.0
    var tpSum = 0.0
    val beta = 0.5 // User specified

    //FIXME - double check updated values

    //val labels = Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    //FIXME - could be made parallel w/udf
    //for (clsIndex <- minLabel to maxLabel - minLabel) {
    for (clsIndex <- labels.indices) {

      val colSum = updatedRows.map(x => x.tail(clsIndex)).sum
      // val rowValueSum = if (classMaps.map(x => x._2).contains(clsIndex)) updatedRows.filter(x => x.head == clsIndex)(0).tail.map(x => x).sum else 0
      val rowValueSum = updatedRows(clsIndex).tail.sum
      println("col sum: " + colSum + " row sum: " + rowValueSum )
      //println("clsIndex: " + clsIndex + " colSum: " + colSum + " rowSum: " + rowValueSum)
      // val tp: Double = if (classMaps.map(x => x._2).contains(clsIndex)) updatedRows.filter(x => x.head == clsIndex)(0).tail(clsIndex) else 0
      val tp: Double = updatedRows(clsIndex).tail(clsIndex)
      val fn: Double = colSum - tp // check
      val fp: Double = rowValueSum - tp // check
      val tn: Double = totalCount - tp - fp - fn

      println(tp + " " + tn + " " + fp + " " + fn)

      val recall = tp / (tp + fn)
      val precision = tp / (tp + fp)



      AvAvg += ((tp + tn) / (tp + tn + fp + fn))
      MAvG *= recall
      RecM += { if(recall.isNaN) 0.0 else recall }
      PrecM += precision

      val getAvFb: Double= {
        val result = ((1 + Math.pow(beta, 2.0)) * precision * recall) / (Math.pow(beta, 2.0) * precision + recall)
        if(result.isNaN) {
          0.0
        }
        else result
      }
      AvFb += getAvFb

      //FIXME - what to do if col/row sum are zero?
      val rowColMaxValue = Math.max(colSum, rowValueSum)
      if(rowColMaxValue > 0) {
        CBA += tp / rowColMaxValue
      }
      else {
        //println("CBA value NaN")
      }

      // for Recu and Precu
      tpSum += tp
      tSum += (tp + fn)
      pSum += (tp + fp)
    }

    AvAvg /= numberOfClasses
    MAvG = {  val result = Math.pow(MAvG, 1/numberOfClasses.toDouble); if(result.isNaN) 0.0 else result } //Math.pow((MAvG), (1/numberOfClasses.toDouble))
    RecM /= numberOfClasses
    PrecM /= numberOfClasses
    Recu = tpSum / tSum
    Precu = tpSum / pSum
    FbM = { val result = ((1 + Math.pow(beta, 2.0)) * PrecM * RecM) / (Math.pow(beta, 2.0) * PrecM + RecM); if(result.isNaN) 0.0 else result }
    Fbu = { val result = ((1 + Math.pow(beta, 2.0)) * Precu * Recu) / (Math.pow(beta, 2.0) * Precu + Recu); if(result.isNaN) 0.0 else result }
    AvFb /= numberOfClasses
    CBA /= numberOfClasses

    Array(AvAvg.toString, MAvG.toString, RecM.toString, PrecM.toString, Recu.toString, Precu.toString, FbM.toString, Fbu.toString, AvFb.toString, CBA.toString)
  }



  import org.apache.spark.ml.knn.{KNN, KNNModel}
  import scala.collection.mutable

  val kValue = 5

  // Map number of nearest same-class neighbors to minority class label
  def getMinorityClassLabel(kCount: Int): Int = {

    if (kCount / kValue.toFloat >= 0.8) { 0 }
    else if ( kCount / kValue.toFloat >= 0.4) { 1 }
    else if ( kCount / kValue.toFloat >= 0.2) { 2 }
    else { 3 }
  }

  def filterByMinorityType(df: DataFrame, minorityLabels: String): DataFrame ={
    val minorityType = Map("S" -> 0, "B" -> 1, "R" -> 2, "O" -> 3)
    val numberValues = minorityLabels.split("").map(x=>minorityType(x))
    df.filter(df("minorityType").isin(numberValues:_*)).select("index", "label", "features")
  }

  /*def getSparkNNMinorityResult(x: mutable.WrappedArray[Any], index: Int, features: Any): (Int, Int, Int, mutable.WrappedArray[Double]) = {
    val wrappedArray = x

    val nearestLabels = Array[Int]()
    def getLabel(neighbor: Any): Int = {
      val index = neighbor.toString.indexOf(",")
      neighbor.toString.substring(1, index).toInt
    }

    val currentLabel = getLabel(wrappedArray(0))
    var currentCount = 0
    for(i<-1 until wrappedArray.length) {
      nearestLabels :+ getLabel(wrappedArray(i))
      if (getLabel(wrappedArray(i)) == currentLabel) {
        currentCount += 1
      }
    }
    val currentArray = features.toString.substring(1, features.toString.length-1).split(",").map(x=>x.toDouble)
    (index, currentLabel, getMinorityClassLabel(currentCount), currentArray)//features.toString().asInstanceOf[mutable.WrappedArray[Double]])
  }

  def getInstanceLevelDifficulty(df: DataFrame, dataFile: String): DataFrame ={
    val path = "/home/ford/data/mt/" + dataFile + "DF"

    val minorityDF = try {
        println("READ KNN")
        val readData = df.sparkSession.read.
          option("inferSchema", true).
          option("header", true).
          csv(path)

        val stringToArray = udf((item: String)=>{
          val features: Array[Double] = item.dropRight(1).drop(1).split(",").map(x=>x.toDouble)
          Vectors.dense(features).toDense
        })

        val intToLong = udf((index: Int)=>{
          index.toLong
        })

        readData.withColumn("features", stringToArray(col("features"))).withColumn("index", intToLong(col("index")))
    } catch {
      case _: Throwable => {
        val t0 = System.nanoTime()
        println("CALCULATE KNN")
        val leafSize = 1000
        val knn = new KNN()
          .setTopTreeSize(df.count.toInt / 10)
          .setTopTreeLeafSize(leafSize)
          .setSubTreeLeafSize(2500)
          .setSeed(42L)
          .setAuxCols(Array("label", "features"))
        val model = knn.fit(df).setK(kValue+1)//.setDistanceCol("distances")
        val results2: DataFrame = model.transform(df)

        val collected: Array[Row] = results2.select( "neighbors", "index", "features").collect()
        val minorityValueDF: Array[(Int, Int, Int, mutable.WrappedArray[Double])] = collected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[Any]],x(1),x(2))).map(x=>getSparkNNMinorityResult(x._1, x._2.toString.toInt, x._3))

        val minorityDF = df.sparkSession.createDataFrame(df.sparkSession.sparkContext.parallelize(minorityValueDF))
          .withColumnRenamed("_1","index")
          .withColumnRenamed("_2","label")
          .withColumnRenamed("_3","minorityType")
          .withColumnRenamed("_4","features").sort("index")

        val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")

        minorityDF.withColumn("features", stringify(col("features"))).
          repartition(1).
          write.format("com.databricks.spark.csv").
          option("header", "true").
          mode("overwrite").
          save(path)
        val t1 = System.nanoTime()
        println("Elapsed time: " + (t1 - t0) / 1e9 + "s")

        minorityDF
      }
    }

    minorityDF
  }*/


  def getSparkNNMinorityResult(neighborLabels: mutable.WrappedArray[String], index: Long, features: DenseVector): (Long, String, Int, DenseVector) = {

    val currentLabel = neighborLabels.array.head
    val nearestLabels = neighborLabels.array.tail

    var currentCount = 0
    for(i<-nearestLabels.indices) {
      if (currentLabel == nearestLabels(i)) {
        currentCount += 1
      }
    }
    // val currentArray = features.toString.substring(1, features.toString.length-1).split(",").map(x=>x.toDouble)
    (index, currentLabel, getMinorityClassLabel(currentCount), features)//features.toString().asInstanceOf[mutable.WrappedArray[Double]])
  }

  def getInstanceLevelDifficulty(df: DataFrame, dataFile: String, minorityTypePath: String): DataFrame ={
    val path = minorityTypePath + "/" + dataFile + "DF"

    val minorityDF = try {
      println("READ KNN")
      val readData = df.sparkSession.read.
        option("inferSchema", true).
        option("header", true).
        csv(path)

      val stringToArray = udf((item: String)=>{
        val features: Array[Double] = item.dropRight(1).drop(1).split(",").map(x=>x.toDouble)
        Vectors.dense(features).toDense
      })

      val intToLong = udf((index: Int)=>{
        index.toLong
      })

      readData.withColumn("features", stringToArray(col("features"))).withColumn("index", intToLong(col("index")))
    } catch {
      case _: Throwable => {
        val spark = df.sparkSession
        import spark.implicits._

        val t0 = System.nanoTime()
        println("CALCULATE KNN")
        val leafSize = 1000
        val knn = new KNN()
          .setTopTreeSize(df.count.toInt / 10)
          .setTopTreeLeafSize(leafSize)
          .setSubTreeLeafSize(2500)
          .setSeed(42L)
          .setAuxCols(Array("label", "features"))
        val model = knn.fit(df).setK(5 + 1)
        val results2: DataFrame = model.transform(df)
        println("at results2")
        results2.printSchema()

        val collected: Array[Row] = results2.withColumn("neighborLabels", $"neighbors.label").select( "neighborLabels", "index", "features").collect()
        val minorityValueDF: Array[(Long, String, Int, DenseVector)] = collected.map(x=>(x(0).asInstanceOf[mutable.WrappedArray[String]], x(1).asInstanceOf[Long], x(2).asInstanceOf[DenseVector])).map(x=>getSparkNNMinorityResult(x._1, x._2.toString.toInt, x._3))


        val minorityDF = df.sparkSession.createDataFrame(df.sparkSession.sparkContext.parallelize(minorityValueDF))
          .withColumnRenamed("_1","index")
          .withColumnRenamed("_2","label")
          .withColumnRenamed("_3","minorityType")
          .withColumnRenamed("_4","features").sort("index")

        println("***** schema")
        minorityDF.printSchema()


        //val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")
        val stringify = udf((v: DenseVector) => {
          v.toString
        })

        //val xx = minorityDF.withColumn("features", stringify(col("features")))
        //println("XX")
        //xx.show()
        //xx.printSchema()


        minorityDF.withColumn("features", stringify(col("features"))).
          repartition(1).
          write.format("com.databricks.spark.csv").
          option("header", "true").
          mode("overwrite").
          save(path)
        val t1 = System.nanoTime()
        println("Elapsed time: " + (t1 - t0) / 1e9 + "s")
        // FIXME - save minority type buld time

        println("READ KNN")
        val readData = df.sparkSession.read.
          option("inferSchema", true).
          option("header", true).
          csv(path)

        val stringToArray = udf((item: String)=>{
          val features: Array[Double] = item.dropRight(1).drop(1).split(",").map(x=>x.toDouble)
          Vectors.dense(features).toDense
        })

        val intToLong = udf((index: Int)=>{
          index.toLong
        })

        readData.withColumn("features", stringToArray(col("features"))).withColumn("index", intToLong(col("index")))
      }
    }

    minorityDF
  }

  def indexDF(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")
  }

  def indexDF2(df: DataFrame): DataFrame ={
    val spark = df.sparkSession
    import spark.implicits._

    val train_index: RDD[(Long, Row)] = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Int, DenseVector)] = train_index.map({ r =>
      val array = r._2.toSeq.toArray//.reverse
      val minorityType = array.head.toString.toInt // array.head.toString.toInt
      val arrayReversed = array.tail.reverse
      val cls = arrayReversed.head.toString
      //val rowMapped: Array[Double] = arrayReversed.tail.head//.map(_.toString.toDouble)
      //println("-- " + arrayReversed.tail.head.asInstanceOf[DenseVector])
     // val x= arrayReversed.tail.map(_.toString)
     // for(v<-x) {
     //   println("~~" + v)
     // }
      //val rowMapped: Array[Double] = array.tail.asInstanceOf[DenseVector].toArray//.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      //println(r._1, r._2)
      (r._1, cls, minorityType, arrayReversed.tail.head.asInstanceOf[DenseVector])
    })

    data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")
  }


  def main(args: Array[String]) {

    val filename = args(0)// "/home/ford/data/sampling_input.txt"
    val lines = Source.fromFile(filename).getLines.map(x=>x.split(":")(0)->x.split(":")(1)).toMap
    val input_file = lines("dataset").trim
    val classifier = lines("classifier").trim
    val samplingMethods = lines("sampling").split(",").map(x=>x.trim)
    val labelColumnName = lines("labelColumn").trim
    val enableDataScaling = if(lines("enableScaling").trim == "true") true else false
    val numSplits = lines("numCrossValidationSplits").trim.toInt
    val minorityTypePath = lines("minorityTypePath").trim
    val savePath = lines("savePath").trim

    println("input")
    for(x<-lines) {
      println(x)
    }
    println("Dataset: " + input_file)
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val df = spark.read.
      option("inferSchema", true).
      option("header", true).
      csv(input_file).withColumnRenamed(labelColumnName, "label")

    /*val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")*/

    val results = indexDF(df)

    results.show()
    results.printSchema()



    val converted: DataFrame = convertFeaturesToVector(results)

    val asDense = udf((v: Any) => {
      if(v.isInstanceOf[SparseVector]) {
        v.asInstanceOf[SparseVector].toDense
      } else {
        v.asInstanceOf[DenseVector]
      }
    })

    val scaledDataIn: DataFrame = if(enableDataScaling) {
      val scaler = new MinMaxScaler()
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.printSchema()
      scaledData.drop("features").withColumn("features", asDense($"scaledFeatures")).drop("scaledFeatures")
    } else { converted }.cache()


    val dataFile = input_file.split("/").reverse.head
    println("***** file : " + dataFile.substring(0, dataFile.length-4))
    val instanceLevelDF = getInstanceLevelDifficulty(scaledDataIn, dataFile.substring(0, dataFile.length-4), minorityTypePath)
    println("@@@@")
    instanceLevelDF.show()

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndexed")

    val datasetIndexed = indexer.fit(instanceLevelDF).transform(instanceLevelDF).drop("label")
      .withColumnRenamed("labelIndexed", "label")

    println("here")
    datasetIndexed.show()
    datasetIndexed.printSchema()

    val scaledData = datasetIndexed//.filter(datasetIndexed("label") === 0.0 || datasetIndexed("label") === 7.0 )


    // FIXME - add pipeline



    val counts = scaledData.count()
    var splits = Array[Int]()
    println("scaled data")
    scaledData.show()
    scaledData.printSchema()

    // splits :+= 2
    splits :+= 0

    if(numSplits < 2) {
      splits :+= (counts * 0.2).toInt
    }
    else {
      for(i <- 1 until numSplits) {
        splits :+= ((counts / numSplits) * i).toInt
      }
    }
    splits :+= counts.toInt

    var combinedSplitResults =  Array[Array[String]]()

    val minorityTypeList = Array("SBRO") //, "SBR", "SBO",  "SRO", "SB", "SR", "SO", "BRO", "BR", "BO", "RO", "S", "B", "R", "O")

    val countsBy = getCountsByClass("label", scaledData)
    countsBy.show
    val labelList = countsBy.select("_1").collect().map(x=>x(0).toString.toDouble)

    val clsFiltered = labelList.map(x=>scaledData.filter(scaledData("label")===x).drop("index")).map(x=>indexDF2(x))

    /*for(x<-clsFiltered) {
      println(x._1 + " " + x._2.count())
    }*/

    def getStratifiedSplit(dfs: Array[DataFrame], totalSplits: Int, splitIndex: Int, minorityType: String = ""): (DataFrame, DataFrame) ={
      val splitIndexLow = (splitIndex)/ totalSplits.toDouble
      val splitIndexHigh = (splitIndex + 1)/ totalSplits.toDouble
      val testFiltered = dfs.map(x => x.filter(x("index") < x.count() * splitIndexHigh && x("index") >= x.count() * splitIndexLow).drop("index"))
      val testDF = testFiltered.reduce(_ union _)

      val trainFiltered = dfs.map(x=>x.filter(x("index") >= x.count() * splitIndexHigh || x("index") < x.count() * splitIndexLow))
      val trainDF = if(minorityType == "" | minorityType == "SBRO") {
        trainFiltered.reduce(_ union _)
      } else {
        filterByMinorityType(trainFiltered.reduce(_ union _), minorityType)
      }
      println("train")
      getCountsByClass("label", trainDF).show()
      println("test")
      getCountsByClass("label", testDF).show()


      (trainDF, testDF)
    }



  // val testData = scaledData.filter(scaledData("index") < splits(splitIndex + 1) && scaledData("index") >= splits(splitIndex)).persist()


    /*println("scaled")
    scaledData.show
    val xx = scaledData.filter(scaledData("label")==="0.0").drop("index")
    xx.show
    val bar = indexDF2(xx)
    bar.show()
    bar.printSchema()*/

    /*clsFiltered(0).show()
    clsFiltered(0).printSchema()

    clsFiltered(1).show()
    clsFiltered(1).printSchema()*/
    //clsFiltered(1).show()


    for(mt<-minorityTypeList) {
      for(splitIndex<-0 until numSplits) {
        println("splitIndex: " + splitIndex + " numSplits: " + numSplits)
        val datasets = if(numSplits == 1) {
          getStratifiedSplit(clsFiltered, 5, 0)
        } else {
          getStratifiedSplit(clsFiltered, numSplits, splitIndex)
        }
        val trainData = datasets._1
        val testData = datasets._2

        //val testData = scaledData.filter(scaledData("index") < splits(splitIndex + 1) && scaledData("index") >= splits(splitIndex)).persist()
        //val trainData = filterByMinorityType(scaledData.filter(scaledData("index") >= splits(splitIndex + 1) || scaledData("index") < splits(splitIndex)), mt).persist()//scaledData.filter(scaledData("index") >= splits(splitIndex + 1) || scaledData("index") < splits(splitIndex)).persist() //filterByMinorityType(scaledData.filter(scaledData("index") >= splits(splitIndex + 1) || scaledData("index") < splits(splitIndex)), "SBRO").persist()
        println("trainSchema")
        trainData.printSchema()

        getCountsByClass("label", trainData).show(100)

        println("original size: " + trainData.count())

        val samplingMap: Map[String, Double] =
          Map( "1" -> 2.0,
            "2" -> 0.5,
            "3" -> 2.0,
            "4" -> 1.0,
            "5" -> 2.0,
            "6" -> 2.0,
            "7" -> 2.0

          /*Map( "none" -> 1.0,
            "rRna" -> 10.0,
            "tRna" -> 20.0,
            "snRna" -> 20.0,
            "mRna" -> 40.0,
            "SRP" -> 40.0,
            "IRES" -> 40.0*/
          )

      // FIXME - make per class map parallel to run faster

       // var resultArray = Array[Array[String]]()
       for(samplingMethod <- samplingMethods) {
         println("$$$$$$$$$$$$$$$$$$$" + samplingMethod + "$$$$$$$$$$$$$$$$$$$")
         val t0 = System.nanoTime()
         val sampledData = if(samplingMethod == "kMeansSmote") {
           val r = new KMeansSMOTE()
           val model = r.fit(trainData).setBalanceThreshold(0.0)//.setTopTreeSize(2)
           model.transform(trainData)
         }  else if(samplingMethod == "borderlineSmote") {
           val r = new BorderlineSMOTE()
           val model = r.fit(trainData).setBalanceThreshold(0.0)
           model.transform(trainData)
         }  else if(samplingMethod == "rbo") {
           val r = new RBO()
           val model = r.fit(trainData)
           model.transform(trainData)
         } else if(samplingMethod == "adasyn") {
           val r = new ADASYN().setBalanceThreshold(0.0)
           val model = r.fit(trainData)
           model.transform(trainData)
         } else if(samplingMethod == "safeLevel") {
           val r = new SafeLevelSMOTE()
           val model = r.fit(trainData)
           model.transform(trainData)
         } else if(samplingMethod == "smote") {
           val r = new SMOTE
           val model = r.fit(trainData).setBalanceThreshold(0.0).setTopTreeSize(2)   //.setSamplingRatios(samplingMap)
           model.transform(trainData)
         } else if(samplingMethod == "mwmote") {
           val r = new MWMOTE()
           val model = r.fit(trainData).setBalanceThreshold(0.0).setTopTreeSize(2)
           model.transform(trainData)
         } else if(samplingMethod == "ccr") {
           val r = new CCR().setEnergy(1.0)
           val model = r.fit(trainData).setBalanceThreshold(0.0)// .setTopTreeSize(10)
           model.transform(trainData)
         } else if(samplingMethod == "ans") {
           val r = new ANS().setdDstanceNeighborLimit(100)
           val model = r.fit(trainData).setBalanceThreshold(0.0)
           model.transform(trainData)
         } else if(samplingMethod == "clusterSmote") {
           val r = new ClusterSMOTE()
           val model = r.fit(trainData).setBalanceThreshold(0.0).setTopTreeSize(2)
           model.transform(trainData)
         } else if(samplingMethod == "gaussianSmote") {
           val r = new GaussianSMOTE()
           val model = r.fit(trainData).setBalanceThreshold(0.0)
           model.transform(trainData)
         } else if(samplingMethod == "smote_d") {
           val r = new SMOTED()
           val model = r.fit(trainData).setBalanceThreshold(0.0)
           model.transform(trainData)
         } else if(samplingMethod == "nras") {
           val r = new NRAS()
           val model = r.fit(trainData)
           model.transform(trainData)
         }  else if(samplingMethod == "randomOversample") {
           val r = new RandomOversample() // multi-class done
           val model = r.fit(trainData)
           model.transform(trainData)
         }  else if(samplingMethod == "randomUndersample") {
           val r = new RandomUndersample() // multi-class done
           val model = r.fit(trainData)
           model.transform(trainData)
         }
         else {
           trainData
         }

         println("new total count: " + sampledData.count())
         getCountsByClass("label", sampledData).show
         sampledData.printSchema()

         val t1 = System.nanoTime()

         val savePathString = savePath
         val saveDirectory = new File(savePathString)
         if (!saveDirectory.exists()) {
           saveDirectory.mkdirs()
         }

         val x: Array[String] = Array(samplingMethod, mt) ++ runClassifierMinorityType(classifier, sampledData, testData) ++ Array(((t1 - t0) / 1e9).toString)

         resultArray = resultArray :+ x
         combinedSplitResults = combinedSplitResults :+ x
       }

       val resultsDF = buildResultDF(spark, resultArray)
       println("Split Number: " + splitIndex)
       resultsDF.show


        // FIXME - save path for method as well
       resultsDF.repartition(1).
         write.format("com.databricks.spark.csv").
         option("header", "true").
         mode("overwrite").
         save(savePath + "/" + splitIndex)

       trainData.unpersist()
       testData.unpersist()
      }
    }

    // println("Total")
    val totalResults = buildResultDF(spark, combinedSplitResults)

    // first("mt").as("mt"),
    val totals = totalResults.groupBy("sampling", "mt").agg(avg("AvAvg").as("AvAvg"),
      avg("MAvG").as("MAvG"), avg("RecM").as("RecM"), avg("Recu").as("Recu"),
      avg("PrecM").as("PrecM"), avg("Precu").as("Precu"), avg("FbM").as("FbM"),
      avg("Fbu").as("Fbu"), avg("AvFb").as("AvFb"), avg("CBA").as("CBA"), avg("time").as("time"))
    totals.show

    totals.repartition(1).
      write.format("com.databricks.spark.csv").
      option("header", "true").
      mode("overwrite").
      save(savePath + "/totals")
  }

  def buildResultDF(spark: SparkSession, resultArray: Array[Array[String]] ): DataFrame = {
    import spark.implicits._

    for(x<-resultArray.indices) {
      for(y<-resultArray(x).indices) {
        // println(y)
      }
    }
    val csvResults = resultArray.map(x => x match {
      case Array(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) => (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12)
    }).toSeq
    val c = spark.sparkContext.parallelize(csvResults).toDF
    val lookup = Map(
      "_1" -> "sampling",
      "_2" -> "mt",
      "_3" -> "AvAvg",
      "_4" -> "MAvG",
      "_5" -> "RecM",
      "_6" -> "Recu",
      "_7" -> "PrecM",
      "_8" -> "Precu",
      "_9" -> "FbM",
      "_10" -> "Fbu",
      "_11" -> "AvFb",
      "_12" -> "CBA",
      "_13" -> "time"
    )

    val cols = c.columns.map(name => lookup.get(name) match {
      case Some(newName) => col(name).as(newName)
      case None => col(name)
    })

    c.select(cols: _*)
  }

  def runClassifierMinorityType(classifier: String, train: DataFrame, test: DataFrame): Array[String] ={

    import train.sparkSession.implicits._
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndexed")


    val indexerModel = indexer.fit(train)
    val indexedTrain = indexerModel.transform(train).withColumnRenamed("label", "originalLabel").withColumnRenamed("labelIndexed", "label")
    val indexedTest = indexerModel.transform(test).withColumnRenamed("label", "originalLabel").withColumnRenamed("labelIndexed", "label")//.sample(0.05)

    val labelMap = indexedTrain.select("originalLabel", "label").distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
    val labelMapReversed = labelMap.map(x=>(x._2, x._1))

    println("^^ runClassifierMinorityType")
    println(labelMapReversed)
    //val spark = train.sparkSession
    //FIXME - don't collect twice
    //val maxLabel: Double = indexedTest.select("label").distinct().collect().map(x => x.toSeq.last.toString.toDouble).max
    //val minLabel: Double = indexedTest.select("label").distinct().collect().map(x => x.toSeq.last.toString.toDouble).min
    // val inputCols = test.columns.filter(_ != "label")

    val cls = if(classifier == "svm") {
      val lsvm = new LinearSVC()
        .setMaxIter(50)
        .setRegParam(0.1)

      new OneVsRest().setClassifier(lsvm)
    } else {
      new RandomForestClassifier().setNumTrees(50).
        setSeed(42L).
        setLabelCol("label").
        setFeaturesCol("features").
        setPredictionCol("prediction")
    }


    indexedTrain.show
    indexedTrain.printSchema()


    val model = cls.fit(indexedTrain)
    val predictions: DataFrame = model.transform(indexedTest)

    predictions.show(100)
    predictions.printSchema()
    //predictions.select("label", "prediction").write.format("csv").option("header", "false").save("/home/ford/tmp/predictions/")

    import org.apache.spark.ml.linalg.Vectors

    //val predictionAndLabels = test.select("label", "prediction").collect().map(x=>(x(0).asInstanceOf[Double], x(1).asInstanceOf[DenseVector])).map { case LabeledPoint(label, features) =>
    /*val predictionAndLabels = test.select("label", "features").map(row =>
      val prediction = model.predict(features)
      //new LabeledPoint(row(0).asInstanceOf[Double],  org.apache.spark.mllib.linalg.Vectors.fromML(row(1).asInstanceOf[DenseVector]))).rdd
      new LabeledPoint(row(0).asInstanceOf[Double],prediction)).rdd
*/

    import org.apache.spark.mllib.linalg.Vectors
    /*val predictionAndLabels = test.select("label", "features").map { row =>
      val f = row(1).asInstanceOf[DenseVector]

      val prediction = model.predict(f)
      (prediction, row(0).asInstanceOf[Double])
    }.rdd


    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("Confusion matrix:")
    println(metrics.confusionMatrix)
    println(metrics)*/

    val labels: Array[Double] = labelMap.values.toArray.sorted

    val confusionMatrix: Dataset[Row] = predictions.groupBy("label").
      pivot("prediction", labels).
      count().
      na.fill(0.0).
      orderBy("label")

    println("cm")
    confusionMatrix.printSchema()

    calculateClassifierResults(indexedTest.select("label").distinct(), confusionMatrix, labels)
  }

}

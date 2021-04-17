package org.apache.spark.ml.sampling

import java.io
import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.{LinearSVC, NaiveBayes, OneVsRest, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, MinMaxScaler, StringIndexer}
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.attribute.Attribute
import scala.io.Source
import org.apache.spark.ml.knn.KNN
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.sampling.utilities.{convertFeaturesToVector, getCountsByClass}
import org.apache.spark.sql.functions._
import org.apache.spark.ml._
import org.apache.spark.sql.types.{DoubleType, StructType}

import scala.collection.mutable
//FIXME - turn classes back to Ints instead of Doubles
object Sampling {

  var resultIndex = 0
  //var results = ""
  var resultArray: Array[Array[String]] = Array()

  // FIXME - define this in another file?
  type NearestClassResult = (Int, Array[Int]) //class, closest classes
  type NearestClassResultIndex = (Long, Int, Array[Int]) //index, class, closest classes
  type Element = (Long, (Int, Array[Float]))
  type DistanceResult = (Float, Int)

  var cParam = 0.0

  def printArray(a: Array[Double]) = {
    for (x <- a) {
      print(x + " ")
    }
    println("")
  }

  def getConfusionMatrix(cm: Array[Array[Double]]): String = {
    var result = ""
    for (row <- cm) {
      for (index <- row.indices) {
        result += row(index)
        if (index < row.length - 1) {
          result += ","
        }
      }
      result += "\n"
    }
    result
  }

  def calculateClassifierResults(distinctClasses: DataFrame, confusionMatrix: DataFrame, labels: Array[Double]): Array[String] = {

    println("LABELS")
    for (x <- labels) {
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
    val classMaps = rows.indices.map(x => (rows(x).head, x)).toMap
    println(classMaps)
    //for(x<-classMaps) {
    //    println(x._1, x._2)
    // }

    var updatedRows = Array[Array[Double]]()
    for (x <- labels.indices) {
      if (classMaps.contains(labels(x))) {
        // println(rows(classMaps(x)).toString)
        //    printArray(rows(classMaps(x)))
        updatedRows = updatedRows :+ rows(classMaps(x))

      } else {
        //updatedRows +: Array[Double](labels(x)) ++ labels.indices.map(_=>0.0)
        val emptyRow = Array[Double](labels(x)) ++ labels.indices.map(_ => 0.0) //, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) // Array[Double](labels(x)) ++ labels.indices.map(_=>0.0)
        // println(emptyRow.toString)
        //  printArray(emptyRow)
        updatedRows = updatedRows :+ emptyRow

      }
    }

    println("updated rows")
    for (x <- updatedRows) {
      printArray(x)
    }


    val totalCount = updatedRows.map(x => x.tail.sum).sum

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
      println("col sum: " + colSum + " row sum: " + rowValueSum)
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
      RecM += {
        if (recall.isNaN) 0.0 else recall
      }
      PrecM += precision

      val getAvFb: Double = {
        val result = ((1 + Math.pow(beta, 2.0)) * precision * recall) / (Math.pow(beta, 2.0) * precision + recall)
        if (result.isNaN) {
          0.0
        }
        else result
      }
      AvFb += getAvFb

      //FIXME - what to do if col/row sum are zero?
      val rowColMaxValue = Math.max(colSum, rowValueSum)
      if (rowColMaxValue > 0) {
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
    MAvG = {
      val result = Math.pow(MAvG, 1 / numberOfClasses.toDouble); if (result.isNaN) 0.0 else result
    } //Math.pow((MAvG), (1/numberOfClasses.toDouble))
    RecM /= numberOfClasses
    PrecM /= numberOfClasses
    Recu = tpSum / tSum
    Precu = tpSum / pSum
    FbM = {
      val result = ((1 + Math.pow(beta, 2.0)) * PrecM * RecM) / (Math.pow(beta, 2.0) * PrecM + RecM); if (result.isNaN) 0.0 else result
    }
    Fbu = {
      val result = ((1 + Math.pow(beta, 2.0)) * Precu * Recu) / (Math.pow(beta, 2.0) * Precu + Recu); if (result.isNaN) 0.0 else result
    }
    AvFb /= numberOfClasses
    CBA /= numberOfClasses

    Array(AvAvg.toString, MAvG.toString, RecM.toString, PrecM.toString, Recu.toString, Precu.toString, FbM.toString, Fbu.toString, AvFb.toString, CBA.toString)
  }


  import org.apache.spark.ml.knn.{KNN, KNNModel}
  import scala.collection.mutable

  val kValue = 5

  // Map number of nearest same-class neighbors to minority class label
  def getMinorityClassLabel(kCount: Int): Int = {

    if (kCount / kValue.toFloat >= 0.8) {
      0
    }
    else if (kCount / kValue.toFloat >= 0.4) {
      1
    }
    else if (kCount / kValue.toFloat >= 0.2) {
      2
    }
    else {
      3
    }
  }

  def filterByMinorityType(df: DataFrame, minorityLabels: String): DataFrame = {
    val minorityType = Map("S" -> 0, "B" -> 1, "R" -> 2, "O" -> 3)
    val numberValues = minorityLabels.split("").map(x => minorityType(x))
    df.filter(df("minorityType").isin(numberValues: _*)).select("index", "label", "features")
  }


  def getSparkNNMinorityResult(neighborLabels: mutable.WrappedArray[String], index: Long, features: DenseVector): (Long, String, Int, DenseVector) = {

    val currentLabel = neighborLabels.array.head
    val nearestLabels = neighborLabels.array.tail

    var currentCount = 0
    for (i <- nearestLabels.indices) {
      if (currentLabel == nearestLabels(i)) {
        currentCount += 1
      }
    }
    // val currentArray = features.toString.substring(1, features.toString.length-1).split(",").map(x=>x.toDouble)
    (index, currentLabel, getMinorityClassLabel(currentCount), features) //features.toString().asInstanceOf[mutable.WrappedArray[Double]])
  }

  def readNonInstanceLevelDiffuclty(df: DataFrame, dataFile: String, minorityTypePath: String): DataFrame = {
    val path = minorityTypePath + "/" + dataFile + "DF"
    val readData = df.sparkSession.read.
      option("inferSchema", true).
      option("header", true).
      csv(path)

    val stringToArray = udf((item: String) => {
      val features: Array[Double] = item.dropRight(1).drop(1).split(",").map(x => x.toDouble)
      Vectors.dense(features).toDense
    })

    val intToLong = udf((index: Int) => {
      index.toLong
    })

    readData.withColumn("features", stringToArray(col("features"))).withColumn("index", intToLong(col("index")))
  }

  def getInstanceLevelDifficulty(df: DataFrame, dataFile: String, minorityTypePath: String): DataFrame = {
    val path = minorityTypePath + "/" + dataFile + "DF"

    val minorityDF = try {
      println("READ KNN")
      val readData = df.sparkSession.read.
        option("inferSchema", true).
        option("header", true).
        csv(path)

      val stringToArray = udf((item: String) => {
        val features: Array[Double] = item.dropRight(1).drop(1).split(",").map(x => x.toDouble)
        Vectors.dense(features).toDense
      })

      val intToLong = udf((index: Int) => {
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

        val collected: Array[Row] = results2.withColumn("neighborLabels", $"neighbors.label").select("neighborLabels", "index", "features").collect()
        val minorityValueDF: Array[(Long, String, Int, DenseVector)] = collected.map(x => (x(0).asInstanceOf[mutable.WrappedArray[String]], x(1).asInstanceOf[Long], x(2).asInstanceOf[DenseVector])).map(x => getSparkNNMinorityResult(x._1, x._2.toString.toInt, x._3))


        val minorityDF = df.sparkSession.createDataFrame(df.sparkSession.sparkContext.parallelize(minorityValueDF))
          .withColumnRenamed("_1", "index")
          .withColumnRenamed("_2", "label")
          .withColumnRenamed("_3", "minorityType")
          .withColumnRenamed("_4", "features").sort("index")

        println("***** schema")
        minorityDF.printSchema()


        //val stringify = udf((vs: Seq[String]) => s"""[${vs.mkString(",")}]""")
        val stringify = udf((v: DenseVector) => {
          v.toString
        })

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

        val stringToArray = udf((item: String) => {
          val features: Array[Double] = item.dropRight(1).drop(1).split(",").map(x => x.toDouble)
          Vectors.dense(features).toDense
        })

        val intToLong = udf((index: Int) => {
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

    val train_index = df.drop("index").rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, Double, DenseVector)] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString.toDouble
      val rowMapped = array.tail(0).asInstanceOf[DenseVector] // Array[Double]()// array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped)
    })

    val indexDF = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")
    indexDF
  }

  def indexDF2(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._

    val train_index: RDD[(Long, Row)] = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Int, DenseVector)] = train_index.map({ r =>
      val array = r._2.toSeq.toArray //.reverse
    val minorityType = array.head.toString.toInt // array.head.toString.toInt
    val arrayReversed = array.tail.reverse
      val cls = arrayReversed.head.toString
      (r._1, cls, minorityType, arrayReversed.tail.head.asInstanceOf[DenseVector])
    })

    data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "minorityType")
      .withColumnRenamed("_4", "features")
  }

  val asDense = udf((v: Any) => {
    if (v.isInstanceOf[SparseVector]) {
      v.asInstanceOf[SparseVector].toDense
    } else {
      v.asInstanceOf[DenseVector]
    }
  })


  def main(args: Array[String]) {

    val filename = args(0) // "/home/ford/data/sampling_input.txt"
    val lines = Source.fromFile(filename).getLines.map(x => x.split(":")(0) -> x.split(":")(1)).toMap
    // val input_file = lines("dataset").trim
    val classifiers = lines("classifier").trim.split(',')
    val samplingMethods = lines("sampling").split(",").map(x => x.trim)
    val labelColumnName = lines("labelColumn").trim
    val enableDataScaling = if (lines("enableScaling").trim == "true") true else false
    val numSplits = lines("numCrossValidationSplits").trim.toInt
    // val minorityTypePath = lines("minorityTypePath").trim
    val savePath = lines("savePath").trim
    val datasetSize = lines("datasetSize").trim
    val datasetPath = lines("datasetPath").trim
    val datasetList = lines("datasetList").trim
    val datasetName = "covtype"//lines("datasetName").trim

    println("***")
    val singleSplitIndex = if (lines.contains("singleSplitIndex")) {
      lines("singleSplitIndex").trim.toInt
    } else {
      -1
    }

    println("singe line split: " + singleSplitIndex)
    val input_file = "/tmp/foo.csv"

    println("input")
    for (x <- lines) {
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

    val train_index = df.rdd.zipWithIndex().map({ case (x, y) => (y, x) }).cache()

    val data: RDD[(Long, String, Array[Double])] = train_index.map({ r =>
      val array = r._2.toSeq.toArray.reverse
      val cls = array.head.toString
      val rowMapped: Array[Double] = array.tail.map(_.toString.toDouble)
      //NOTE - This needs to be back in the original order to train/test works correctly
      (r._1, cls, rowMapped.reverse)
    })

    val results: DataFrame = data.toDF().withColumnRenamed("_1", "index")
      .withColumnRenamed("_2", "label")
      .withColumnRenamed("_3", "features")

    val converted: DataFrame = convertFeaturesToVector(results)

    val scaledDataIn: DataFrame = if (enableDataScaling) {
      val scaler = new MinMaxScaler().setMin(0.0).setMax(1.0)
        .setInputCol("features")
        .setOutputCol("scaledFeatures")
      val scalerModel = scaler.fit(converted)
      val scaledData: DataFrame = scalerModel.transform(converted)
      scaledData.printSchema()
      scaledData.drop("features").withColumn("features", asDense($"scaledFeatures")).drop("scaledFeatures")
    } else {
      converted
    }.cache()

    scaledDataIn.show()
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndexed")


    val indexerModel = indexer.fit(scaledDataIn)
    val datasetIndexed = indexerModel.transform(scaledDataIn).drop("label")
      .withColumnRenamed("labelIndexed", "label")



    val trainData = datasetIndexed

    val countsBy = getCountsByClass("label", trainData)
    countsBy.show


    for (samplingMethod <- samplingMethods) {
      println("$$$$$$$$$$$$$$$$$$$" + samplingMethod + "$$$$$$$$$$$$$$$$$$$")
      val sampledData = if (samplingMethod == "kMeansSmote") {
        val r = new KMeansSMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0) //.setTopTreeSize(2)
        model.transform(trainData)
      } else if (samplingMethod == "borderlineSmote") {
        val r = new BorderlineSMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0)
        model.transform(trainData)
      } else if (samplingMethod == "rbo") {
        val r = new RBO().setGamma(0.001).setStepSize(0.001).setIterations(100).setStoppingProbability(0.001)
        val model = r.fit(trainData)
        model.transform(trainData)
      } else if (samplingMethod == "adasyn") {
        val r = new ADASYN().setBalanceThreshold(0.0)
        val model = r.fit(trainData)
        model.transform(trainData)
      } else if (samplingMethod == "safeLevel") {
        val r = new SafeLevelSMOTE()
        val model = r.fit(trainData)
        model.transform(trainData)
      } else if (samplingMethod == "smote") {
        val r = new SMOTE
        val model = r.fit(trainData).setBalanceThreshold(0.0) //.setTopTreeSize(2)   //.setSamplingRatios(samplingMap)
        model.transform(trainData)
      } else if (samplingMethod == "mwmote") {
        val r = new MWMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0) //.setTopTreeSize(2)
        model.transform(trainData)
      } else if (samplingMethod == "ccr") {
        val r = new CCR().setEnergy(1.0)
        val model = r.fit(trainData).setBalanceThreshold(0.0) // .setTopTreeSize(10)
        model.transform(trainData)
      } else if (samplingMethod == "ans") {
        val r = new ANS().setdDstanceNeighborLimit(100)
        val model = r.fit(trainData).setBalanceThreshold(0.0)
        model.transform(trainData)
      } else if (samplingMethod == "clusterSmote") {
        val r = new ClusterSMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0) //.setTopTreeSize(2)
        model.transform(trainData)
      } else if (samplingMethod == "gaussianSmote") {
        val r = new GaussianSMOTE()
        val model = r.fit(trainData).setBalanceThreshold(0.0)
        model.transform(trainData)
      } else if (samplingMethod == "smote_d") {
        val r = new SMOTED()
        val model = r.fit(trainData).setBalanceThreshold(0.0)
        model.transform(trainData)
      } else if (samplingMethod == "nras") {
        val r = new NRAS()
        val model = r.fit(trainData)
        model.transform(trainData)
      } else if (samplingMethod == "randomOversample") {
        val r = new RandomOversample() // multi-class done
        val model = r.fit(trainData)
        model.transform(trainData)
      } else if (samplingMethod == "randomUndersample") {
        val r = new RandomUndersample() // multi-class done
        val model = r.fit(trainData)
        model.transform(trainData)
      }
      else {
        trainData
      }

      val sampledCounts = getCountsByClass("label", sampledData)
      sampledCounts.show
      sampledCounts.printSchema()

      sampledData.show
      sampledData.printSchema()



      // println(indexerModel.labelsArray)
      /*for(x<-indexerModel.labelsArray) {
        for(y<-x) {
          print(y + " " )
        }
        println()
      }*/

      /*val df = spark.createDataFrame(Seq(
        (0.0, "a"),
        (1.0, "b"),
        (2.0, "c"),
        (3.0, "a"),
        (4.0, "a"),
        (5.0, "c")
      )).toDF("category", "label")
      print("** original")
      df.show

      val indexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("labelIndex")
        .fit(df)
      val indexed = indexer.transform(df)

      println(s"Transformed string column '${indexer.getInputCol}' " +
        s"to indexed column '${indexer.getOutputCol}'")
      println("*** indexed")
      indexed.show()


      val temp = indexed.drop(col("label")).withColumn("label", col("labelIndex")).drop("labelIndex")
      println("*** temp")
      temp.show

      val converter = new IndexToString()
        .setInputCol("label")
        .setOutputCol("label2")

      val converted = converter.transform(temp).drop("label").withColumn("label", col("label2")).drop("label2")
      println("*** converted")
      converted.show*/
    import spark.implicits._
      import org.apache.spark.sql.types._
    val foo = sampledData.withColumn("label2", $"label".cast("double")).drop("label").withColumn("label", col("label2")).drop("label2")
      foo.show()
    foo.printSchema()
      println("************************")
      println(foo.take(1)(0))
      val converter2 = new IndexToString()
        .setInputCol("label")
        .setOutputCol("label2")

      // val converted2 = converter2.transform(foo).drop("label").withColumn("label", col("label2")).drop("label2")
      val convertedX = converter2.transform(foo)//.drop("label").withColumn("label", col("label2")).drop("label2")
     println("*** converted")
     convertedX.show
     val converted2 = sampledData

      def vecToArray(row: Row): Array[Double] ={
        row(0).asInstanceOf[DenseVector].toArray ++ Array(row(1).toString.toDouble)//.asInstanceOf[Double])
      }

      val featuresDF = converted2//.select("features")
      featuresDF.show
      featuresDF.printSchema()
      println(featuresDF.take(1)(0))
      val collected = featuresDF.collect().map(x=>vecToArray(x))
      val rows: Array[Row] = collected.map{ x => Row(x:_*)}
      println(rows(0))

      import org.apache.spark.sql.types._
      import org.apache.spark.sql.functions._
      import spark.sqlContext.implicits._
      import spark.implicits._
      val d: RDD[Row] = featuresDF.sparkSession.sparkContext.parallelize(rows)
      val schema = new StructType( Array(
        StructField("x", DoubleType, nullable=false),
        StructField("y", DoubleType, nullable=false),
        StructField("label", DoubleType, nullable=false)))

      val result = featuresDF.sparkSession.createDataFrame(d, schema)
      result.show

    println("save path: " + savePath)
      result.repartition(1).
        write.format("com.databricks.spark.csv").
        option("header", "true").
        mode("overwrite").
        save(savePath)

      //
      //val data = Seq(("Java", "20000"), ("Python", "100000"), ("Scala", "3000"))
      //val dfFromData1 = data.toDF()
      //dfFromData1.show

      /*sampledData.repartition(1).
        write.format("com.databricks.spark.csv").
        option("header", "true").
        mode("overwrite").
        save("/tmp/data")*/
    }

    //val labelList = countsBy.select("_1").collect().map(x => x(0).toString.toDouble)
    //val clsFiltered = labelList.map(x => scaledData.filter(scaledData("label") === x))

    /*def getStratifiedSplit(dfs: Array[DataFrame], totalSplits: Int, splitIndex: Int, minorityType: String = ""): (DataFrame, DataFrame) = {
      val splitIndexLow = splitIndex / totalSplits.toDouble
      val splitIndexHigh = (splitIndex + 1) / totalSplits.toDouble

      println("*** split index ***")
      println(splitIndex + " " + totalSplits.toDouble)
      println("******")

      val testFiltered = dfs.map(x => indexDF(x)).map(x => x.filter(x("index") < x.count() * splitIndexHigh && x("index") >= x.count() * splitIndexLow))
      val testDF = testFiltered.reduce(_ union _)

      val trainFiltered = dfs.map(x => indexDF(x)).map(x => x.filter(x("index") >= x.count() * splitIndexHigh || x("index") < x.count() * splitIndexLow))
      val trainDF = trainFiltered.reduce(_ union _)

      println("*train")
      getCountsByClass("label", trainDF).show()
      //println(trainDF.count())
      //trainDF.printSchema()

      println("*test")
      getCountsByClass("label", testDF).sort(asc("_1")).show(100)
      println(testDF.count())

      //val testX: Array[DataFrame] = dfs.map(x => indexDF(x))

      //println("~~~~ class data")
      //testX(0).show(100)
      //println(splitIndexLow + " " + splitIndexHigh)
      //testFiltered(0).show(100)

      //println("~~~~ end class data")

      (trainDF, testDF)
    }*/

    /*for (splitIndex <- 0 until numSplits) {
      resultArray = Array()
      println("splitIndex: " + splitIndex + " numSplits: " + numSplits)
      val datasets = if (numSplits == 1) {
        getStratifiedSplit(clsFiltered, 5, singleSplitIndex)
      } else {
        getStratifiedSplit(clsFiltered, numSplits, splitIndex)
      }
      val trainData = datasets._1
      val testData = datasets._2

      println("trainSchema")
      trainData.printSchema()

      println("train")
      getCountsByClass("label", trainData).show(100)

      println("test")
      getCountsByClass("label", testData).show(100)

      println("original size: " + trainData.count())

      for (samplingMethod <- samplingMethods) {
        println("$$$$$$$$$$$$$$$$$$$" + samplingMethod + "$$$$$$$$$$$$$$$$$$$")
        val t0 = System.nanoTime()
        val sampledData = if (samplingMethod == "kMeansSmote") {
          val r = new KMeansSMOTE()
          val model = r.fit(trainData).setBalanceThreshold(0.0) //.setTopTreeSize(2)
          model.transform(trainData)
        } else if (samplingMethod == "borderlineSmote") {
          val r = new BorderlineSMOTE()
          val model = r.fit(trainData).setBalanceThreshold(0.0)
          model.transform(trainData)
        } else if (samplingMethod == "rbo") {
          val r = new RBO().setGamma(0.001).setStepSize(0.001).setIterations(100).setStoppingProbability(0.001)
          val model = r.fit(trainData)
          model.transform(trainData)
        } else if (samplingMethod == "adasyn") {
          val r = new ADASYN().setBalanceThreshold(0.0)
          val model = r.fit(trainData)
          model.transform(trainData)
        } else if (samplingMethod == "safeLevel") {
          val r = new SafeLevelSMOTE()
          val model = r.fit(trainData)
          model.transform(trainData)
        } else if (samplingMethod == "smote") {
          val r = new SMOTE
          val model = r.fit(trainData).setBalanceThreshold(0.0) //.setTopTreeSize(2)   //.setSamplingRatios(samplingMap)
          model.transform(trainData)
        } else if (samplingMethod == "mwmote") {
          val r = new MWMOTE()
          val model = r.fit(trainData).setBalanceThreshold(0.0) //.setTopTreeSize(2)
          model.transform(trainData)
        } else if (samplingMethod == "ccr") {
          val r = new CCR().setEnergy(1.0)
          val model = r.fit(trainData).setBalanceThreshold(0.0) // .setTopTreeSize(10)
          model.transform(trainData)
        } else if (samplingMethod == "ans") {
          val r = new ANS().setdDstanceNeighborLimit(100)
          val model = r.fit(trainData).setBalanceThreshold(0.0)
          model.transform(trainData)
        } else if (samplingMethod == "clusterSmote") {
          val r = new ClusterSMOTE()
          val model = r.fit(trainData).setBalanceThreshold(0.0) //.setTopTreeSize(2)
          model.transform(trainData)
        } else if (samplingMethod == "gaussianSmote") {
          val r = new GaussianSMOTE()
          val model = r.fit(trainData).setBalanceThreshold(0.0)
          model.transform(trainData)
        } else if (samplingMethod == "smote_d") {
          val r = new SMOTED()
          val model = r.fit(trainData).setBalanceThreshold(0.0)
          model.transform(trainData)
        } else if (samplingMethod == "nras") {
          val r = new NRAS()
          val model = r.fit(trainData)
          model.transform(trainData)
        } else if (samplingMethod == "randomOversample") {
          val r = new RandomOversample() // multi-class done
          val model = r.fit(trainData)
          model.transform(trainData)
        } else if (samplingMethod == "randomUndersample") {
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
      }*/

    // resultArray = resultArray.map(x=>x ++ Array((x.tail.reverse(0).toDouble + x.tail.reverse(1).toDouble).toString))
    /*val resultsDF = buildResultDF(spark, resultArray)
      println("Split Number: " + splitIndex)
      resultsDF.show

      // FIXME - save path for method as well
      resultsDF.repartition(1).
        write.format("com.databricks.spark.csv").
        option("header", "true").
        mode("overwrite").
        save(savePath + "/" + datasetName + datasetSize + "/" + splitIndex)

      trainData.unpersist()
      testData.unpersist()*/

  }

  /*val totalResults = buildResultDF(spark, combinedSplitResults)

    val totals = totalResults.groupBy("dataset", "sampling", "classifier").agg(avg("AvAvg").as("AvAvg"),
      avg("MAvG").as("MAvG"), avg("RecM").as("RecM"),
      avg("Recu").as("Recu"), avg("PrecM").as("PrecM"),
      avg("Precu").as("Precu"), avg("FbM").as("FbM"),
      avg("Fbu").as("Fbu"), avg("AvFb").as("AvFb"),
      avg("CBA").as("CBA"), avg("classifierTime").as("classifierTime"),
      avg("samplingTime").as("samplingTime"), avg("totalTime").as("totalTime"))
    totals.show

    totals.repartition(1).
      write.format("com.databricks.spark.csv").
      option("header", "true").
      mode("overwrite").
      save(savePath + "/" + datasetName + datasetSize + "/totals")
  */

//  }

  def buildResultDF(spark: SparkSession, resultArray: Array[Array[String]] ): DataFrame = {
    import spark.implicits._

    for(x<-resultArray.indices) {
      for(y<-resultArray(x).indices) {
        // println(y)
      }
    }
    val csvResults = resultArray.map(x => x match {
      case Array(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15) => (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15)
    }).toSeq
    val c = spark.sparkContext.parallelize(csvResults).toDF
    val lookup = Map(
      "_1" -> "dataset",
      "_2" -> "sampling",
      "_3" -> "classifier",
      "_4" -> "AvAvg",
      "_5" -> "MAvG",
      "_6" -> "RecM",
      "_7" -> "Recu",
      "_8" -> "PrecM",
      "_9" -> "Precu",
      "_10" -> "FbM",
      "_11" -> "Fbu",
      "_12" -> "AvFb",
      "_13" -> "CBA",
      "_14" -> "classifierTime",
      "_15" -> "samplingTime",
      "_16" -> "totalTime"
    )

    val cols = c.columns.map(name => lookup.get(name) match {
      case Some(newName) => col(name).as(newName)
      case None => col(name)
    })

    c.select(cols: _*)
  }

  def runClassifierMinorityType(classifiers: Array[String], train: DataFrame, test: DataFrame): Array[Array[String]] ={
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndexed")

    val indexerModel = indexer.fit(train)
    val indexedTrain = indexerModel.transform(train).withColumnRenamed("label", "originalLabel").withColumnRenamed("labelIndexed", "label").repartition(32)
    val indexedTest = indexerModel.transform(test).withColumnRenamed("label", "originalLabel").withColumnRenamed("labelIndexed", "label").repartition(32)

    val labelMap = indexedTrain.select("originalLabel", "label").distinct().collect().map(x=>(x(0).toString, x(1).toString.toDouble)).toMap
    val labelMapReversed = labelMap.map(x=>(x._2, x._1))

    println("^^ runClassifierMinorityType")
    println(labelMapReversed)

    train.show()

    val labels: Array[Double] = labelMap.values.toArray.sorted


    var results = Array[Array[String]]()
 // FIXME - Check parallelism
    for(classifier<-classifiers) {
      val t0 = System.nanoTime()
      val cls = if(classifier == "svm") {
        val lsvm = new LinearSVC()
          .setMaxIter(100)
          .setRegParam(10.0)

        new OneVsRest().setClassifier(lsvm)
      } else if(classifier == "nb") {
        new NaiveBayes()
      } else {
        new RandomForestClassifier().setNumTrees(100).setMaxDepth(20).
          setSeed(42L).
          setLabelCol("label").
          setFeaturesCol("features").
          setPredictionCol("prediction")
      }

      indexedTrain.show
      indexedTrain.printSchema()
      println("^^^^^ num partitions:" + indexedTrain.rdd.getNumPartitions)

      val model = cls.fit(indexedTrain)
      println("model: " + model.uid)
      val predictions: DataFrame = model.transform(indexedTest)

      val t1 = System.nanoTime()
      val classifierTime = (t1 - t0) / 1e9

      predictions.show(100)
      predictions.printSchema()
      println("prediction partitions: " + predictions.rdd.getNumPartitions)

      val confusionMatrix: Dataset[Row] = predictions.groupBy("label").
        pivot("prediction", labels).
        count().
        na.fill(0.0).
        orderBy("label")

      println("cm")
      confusionMatrix.printSchema()

      results :+= Array(classifier) ++ calculateClassifierResults(indexedTest.select("label").distinct(), confusionMatrix, labels) ++ Array(classifierTime.toString)

    }

    results
  }

}

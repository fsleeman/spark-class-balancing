name := "spark-sampling"

version := "0.1"

// scalaVersion := "2.11.12"
scalaVersion := "2.12.12"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.0"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.0"

libraryDependencies += "org.scalanlp" %% "breeze-natives" % "1.0"

libraryDependencies += "org.scalanlp" %% "breeze" % "1.0"

// libraryDependencies += "com.github.fommil.netlib" % "all" % "1.1.2" // pomOnly()

resolvers += "Spark Packages Repo" at "http://dl.bintray.com/spark-packages/maven"

// libraryDependencies += "saurfang" % "spark-knn" % "0.2.0"

libraryDependencies += "org.apache.spark.ml,knn" % "spark-knn" % "0.2.0" from "file:////home/ford/repos/fsleeman/spark-knn/classes/artifacts/spark_knn_core_spark_knn_jar/spark-knn-core-spark-knn.jar"

mainClass in assembly := Some("edu.vcu.sleeman.Sampling")



assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
 // case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
}

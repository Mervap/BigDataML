name := "BigDataMl"

version := "0.1"

scalaVersion := "2.12.12"

val sparkVersion = "3.2.1"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion withSources(),
  "org.apache.spark" %% "spark-mllib" % sparkVersion withSources()
)

libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.12" % "test" withSources())

Compile / scalaSource := baseDirectory.value / "src/main"
Test / scalaSource := baseDirectory.value / "src/test"

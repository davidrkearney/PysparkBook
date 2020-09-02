# Pyspark Regression with Fiscal Data

> "A minimal example of using Pyspark for Linear Regression"

- toc: true- branch: master- badges: true
- comments: true
- author: David Kearney
- categories: [pyspark, jupyter]
- description: A minimal example of using Pyspark for Linear Regression
- title: Pyspark Regression with Fiscal Data

## Bring in needed imports

from pyspark.sql.functions import col
from pyspark.sql.types import StringType,BooleanType,DateType,IntegerType
from pyspark.sql.functions import *

## Load data from CSV

#collapse-hide

# Load data from a CSV
file_location = "/FileStore/tables/df_panel_fix.csv"
df = spark.read.format("CSV").option("inferSchema", True).option("header", True).load(file_location)
display(df.take(5))


df.createOrReplaceTempView("fiscal_stats")

sums = spark.sql("""
select year, sum(it) as total_yearly_it, sum(fr) as total_yearly_fr
from fiscal_stats
group by 1
order by year asc
""")

sums.show()

## Describing the Data

df.describe().toPandas().transpose()


## Cast Data Type

df2 = df.withColumn("gdp",col("gdp").cast(IntegerType())) \
.withColumn("specific",col("specific").cast(IntegerType())) \
.withColumn("general",col("general").cast(IntegerType())) \
.withColumn("year",col("year").cast(IntegerType())) \
.withColumn("fdi",col("fdi").cast(IntegerType())) \
.withColumn("rnr",col("rnr").cast(IntegerType())) \
.withColumn("rr",col("rr").cast(IntegerType())) \
.withColumn("i",col("i").cast(IntegerType())) \
.withColumn("fr",col("fr").cast(IntegerType()))

## printSchema

df2.printSchema()

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

assembler = VectorAssembler(inputCols=['gdp', 'fdi'], outputCol="features")
train_df = assembler.transform(df2) 

train_df.select("specific", "year").show()

## Linear Regression in Pyspark

lr = LinearRegression(featuresCol = 'features', labelCol='it')
lr_model = lr.fit(train_df)

trainingSummary = lr_model.summary
print("Coefficients: " + str(lr_model.coefficients))
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("R2: %f" % trainingSummary.r2)


lr_predictions = lr_model.transform(train_df)
lr_predictions.select("prediction","it","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="it",metricName="r2")



print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()

predictions = lr_model.transform(test_df)
predictions.select("prediction","it","features").show()

from pyspark.ml.regression import DecisionTreeRegressor
dt = DecisionTreeRegressor(featuresCol ='features', labelCol = 'it')
dt_model = dt.fit(train_df)
dt_predictions = dt_model.transform(train_df)
dt_evaluator = RegressionEvaluator(
    labelCol="it", predictionCol="prediction", metricName="rmse")
rmse = dt_evaluator.evaluate(dt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)


from pyspark.ml.regression import GBTRegressor
gbt = GBTRegressor(featuresCol = 'features', labelCol = 'it', maxIter=10)
gbt_model = gbt.fit(train_df)
gbt_predictions = gbt_model.transform(train_df)
gbt_predictions.select('prediction', 'it', 'features').show(5)


gbt_evaluator = RegressionEvaluator(
    labelCol="it", predictionCol="prediction", metricName="rmse")
rmse = gbt_evaluator.evaluate(gbt_predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

This post includes code adapted from [Spark and Python for Big Data udemy course](https://udemy.com/course/spark-and-python-for-big-data-with-pyspark) and [Spark and Python for Big Data notebooks](https://github.com/SuperJohn/spark-and-python-for-big-data-with-pyspark).
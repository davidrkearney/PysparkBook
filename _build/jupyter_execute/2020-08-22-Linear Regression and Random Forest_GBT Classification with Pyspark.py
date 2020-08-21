# Linear Regression and Random Forest/GBT Classification with Pyspark

data_schema = [
StructField("_c0", IntegerType(), True)
,StructField("province", StringType(), True)
,StructField("specific", DoubleType(), True)
,StructField("general", DoubleType(), True)
,StructField("year", IntegerType(), True)
,StructField("gdp", FloatType(), True)
,StructField("fdi", FloatType(), True)
,StructField("rnr", DoubleType(), True)
,StructField("rr", FloatType(), True)
,StructField("i", FloatType(), True)
,StructField("fr", IntegerType(), True)
,StructField("reg", StringType(), True)
,StructField("it", IntegerType(), True)
]

final_struc = StructType(fields=data_schema)

file_location = "/FileStore/tables/df_panel_fix.csv"
df = spark.read.format("CSV").schema(final_struc).option("header", True).load(file_location)

#df.printSchema()

df.show()

df.groupBy('province').count().show()

mean_val = df.select(mean(df['general'])).collect()
mean_val[0][0]
mean_gen = mean_val[0][0]
df = df.na.fill(mean_gen,["general"])

mean_val = df.select(mean(df['specific'])).collect()
mean_val[0][0]
mean_gen = mean_val[0][0]
df = df.na.fill(mean_gen,["specific"])

mean_val = df.select(mean(df['rr'])).collect()
mean_val[0][0]
mean_gen = mean_val[0][0]
df = df.na.fill(mean_gen,["rr"])

mean_val = df.select(mean(df['fr'])).collect()
mean_val[0][0]
mean_gen = mean_val[0][0]
df = df.na.fill(mean_gen,["fr"])

mean_val = df.select(mean(df['rnr'])).collect()
mean_val[0][0]
mean_gen = mean_val[0][0]
df = df.na.fill(mean_gen,["rnr"])

mean_val = df.select(mean(df['i'])).collect()
mean_val[0][0]
mean_gen = mean_val[0][0]
df = df.na.fill(mean_gen,["i"])

from pyspark.sql.functions import *
df = df.withColumn('specific_classification',when(df.specific >= 583470.7303370787,1).otherwise(0))

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="province", outputCol="provinceIndex")
df = indexer.fit(df).transform(df)

indexer = StringIndexer(inputCol="reg", outputCol="regionIndex")
df = indexer.fit(df).transform(df)

df.show()

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

df.columns

assembler = VectorAssembler(
 inputCols=[
 'provinceIndex',
# 'specific',
 'general',
 'year',
 'gdp',
 'fdi',
 #'rnr',
 #'rr',
 #'i',
 #'fr',
 'regionIndex',
 'it'
 ],
 outputCol="features")

output = assembler.transform(df)

final_data = output.select("features", "specific")

train_data,test_data = final_data.randomSplit([0.7,0.3])

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol='specific')

lrModel = lr.fit(train_data)

print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))

test_results = lrModel.evaluate(test_data)

print("RMSE: {}".format(test_results.rootMeanSquaredError))
print("MSE: {}".format(test_results.meanSquaredError))
print("R2: {}".format(test_results.r2))

from pyspark.sql.functions import corr

df.select(corr('specific','gdp')).show()

from pyspark.ml.classification import DecisionTreeClassifier,GBTClassifier,RandomForestClassifier
from pyspark.ml import Pipeline

dtc = DecisionTreeClassifier(labelCol='specific_classification',featuresCol='features')
rfc = RandomForestClassifier(labelCol='specific_classification',featuresCol='features')
gbt = GBTClassifier(labelCol='specific_classification',featuresCol='features')

final_data = output.select("features", "specific_classification")
train_data,test_data = final_data.randomSplit([0.7,0.3])

rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)
dtc_model = dtc.fit(train_data)

dtc_predictions = dtc_model.transform(test_data)
rfc_predictions = rfc_model.transform(test_data)
gbt_predictions = gbt_model.transform(test_data)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_evaluator = MulticlassClassificationEvaluator(labelCol="specific_classification", predictionCol="prediction", metricName="accuracy")

dtc_acc = acc_evaluator.evaluate(dtc_predictions)
rfc_acc = acc_evaluator.evaluate(rfc_predictions)
gbt_acc = acc_evaluator.evaluate(gbt_predictions)

print('-'*80)
print('Decision tree accuracy: {0:2.2f}%'.format(dtc_acc*100))
print('-'*80)
print('Random forest ensemble accuracy: {0:2.2f}%'.format(rfc_acc*100))
print('-'*80)
print('GBT accuracy: {0:2.2f}%'.format(gbt_acc*100))
print('-'*80)

df.select(corr('specific_classification','fdi')).show()

df.select(corr('specific_classification','gdp')).show()
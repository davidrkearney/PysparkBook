# Window functions and Pivot Tables with Pyspark

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

df.limit(10).toPandas()

df = df.withColumnRenamed("reg","region")

df.limit(10).toPandas()

# df = df.toDF(*['year', 'region', 'province', 'gdp', 'fdi', 'specific', 'general', 'it', 'fr', 'rnr', 'rr', 'i', '_c0', 'specific_classification', 'provinceIndex', 'regionIndex'])

df = df.select('year','region','province','gdp', 'fdi')

df.sort("gdp").show()

from pyspark.sql import functions as F
df.sort(F.desc("gdp")).show()

from pyspark.sql.types import IntegerType, StringType, DoubleType
df = df.withColumn('gdp', F.col('gdp').cast(DoubleType()))

df = df.withColumn('province', F.col('province').cast(StringType()))

df.filter((df.gdp>10000) & (df.region=='East China')).show()

from pyspark.sql import functions as F

df.groupBy(["region","province"]).agg(F.sum("gdp") ,F.max("gdp")).show()

df.groupBy(["region","province"]).agg(F.sum("gdp").alias("SumGDP"),F.max("gdp").alias("MaxGDP")).show()

df.groupBy(["region","province"]).agg(
    F.sum("gdp").alias("SumGDP"),\
    F.max("gdp").alias("MaxGDP")\
    ).show()

df.limit(10).toPandas()

casesWithNewConfirmed = cases.withColumn("NewConfirmed", 100 + F.col("confirmed"))
casesWithNewConfirmed.show()

df = df.withColumn("Exp_GDP", F.exp("gdp"))
df.show()

> Note: Window functions

# Window functions

from pyspark.sql.window import Window
windowSpec = Window().partitionBy(['province']).orderBy(F.desc('gdp'))
df.withColumn("rank",F.rank().over(windowSpec)).show()

from pyspark.sql.window import Window
windowSpec = Window().partitionBy(['province']).orderBy('year')

dfWithLag = df.withColumn("lag_7",F.lag("gdp", 7).over(windowSpec))

df.filter(df.year>'2000').show()

from pyspark.sql.window import Window

windowSpec = Window().partitionBy(['province']).orderBy('year').rowsBetween(-6,0)

dfWithRoll = df.withColumn("roll_7_confirmed",F.mean("gdp").over(windowSpec))

dfWithRoll.filter(dfWithLag.year>'2001').show()

from pyspark.sql.window import Window
windowSpec = Window().partitionBy(['province']).orderBy('year').rowsBetween(Window.unboundedPreceding,Window.currentRow)


dfWithRoll = df.withColumn("cumulative_gdp",F.sum("gdp").over(windowSpec))

dfWithRoll.filter(dfWithLag.year>'1999').show()

> Note: Pivot Dataframes

pivoted_df = df.groupBy('year').pivot('province') \
                      .agg(F.sum('gdp').alias('gdp') , F.sum('fdi').alias('fdi'))
pivoted_df.limit(10).toPandas()

pivoted_df.columns

newColnames = [x.replace("-","_") for x in pivoted_df.columns]

pivoted_df = pivoted_df.toDF(*newColnames)

expression = ""
cnt=0
for column in pivoted_df.columns:
    if column!='year':
        cnt +=1
        expression += f"'{column}' , {column},"
        
expression = f"stack({cnt}, {expression[:-1]}) as (Type,Value)"

unpivoted_df = pivoted_df.select('year',F.expr(expression))
unpivoted_df.show()


```{toctree}
:hidden:
:titlesonly:
:numbered: 

2020-08-21-RDDs and Schemas and Data Types with Pyspark
intro
markdown
2020-08-19-Pyspark-Filtering
2020-08-22-Linear Regression and Random Forest_GBT Classification with Pyspark
2020-08-15-Pyspark-Fiscal-Data-Regression
2020-08-20-Pyspark-Dataframes-Data-Types
2020-08-18-Pyspark-NAs
notebooks
content
```

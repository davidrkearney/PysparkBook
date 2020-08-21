# RDDs and Schemas and Data Types with Pyspark

# Load data from a CSV
file_location = "/FileStore/tables/df_panel_fix.csv"
df = spark.read.format("CSV").option("inferSchema", True).option("header", True).load(file_location)
display(df.take(5))

df.show()

df.printSchema()

df.columns

df.describe()

from pyspark.sql.types import StructField,StringType,IntegerType,StructType

data_schema = [
StructField("_c0", IntegerType(), True)
,StructField("province", StringType(), True)
,StructField("specific", IntegerType(), True)
,StructField("general", IntegerType(), True)
,StructField("year", IntegerType(), True)
,StructField("gdp", IntegerType(), True)
,StructField("fdi", IntegerType(), True)
,StructField("rnr", IntegerType(), True)
,StructField("rr", IntegerType(), True)
,StructField("i", IntegerType(), True)
,StructField("fr", IntegerType(), True)
,StructField("reg", StringType(), True)
,StructField("it", IntegerType(), True)
]


final_struc = StructType(fields=data_schema)

df = spark.read.format("CSV").schema(final_struc).load(file_location)

df.printSchema()

df.show()

df['fr']

type(df['fr'])

df.select('fr')

type(df.select('fr'))

df.select('fr').show()

df.head(2)

df.select(['reg','fr'])

df.select(['reg','fr']).show()

df.withColumn('fiscal_revenue',df['fr']).show()

df.show()

df.withColumnRenamed('fr','new_fiscal_revenue').show()

df.withColumn('double_fiscal_revenue',df['fr']*2).show()

df.withColumn('add_fiscal_revenue',df['fr']+1).show()

df.withColumn('half_fiscal_revenue',df['fr']/2).show()


df.withColumn('half_fr',df['fr']/2)

df.createOrReplaceTempView("economic_data")

sql_results = spark.sql("SELECT * FROM economic_data")

sql_results

sql_results.show()

spark.sql("SELECT * FROM economic_data WHERE fr=634562").show()
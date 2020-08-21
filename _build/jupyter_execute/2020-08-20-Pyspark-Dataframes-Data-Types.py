# Dataframes, Formatting, Casting Data Type and Correlation with Pyspark

# Load data from a CSV
file_location = "/FileStore/tables/df_panel_fix.csv"
df = spark.read.format("CSV").option("inferSchema", True).option("header", True).load(file_location)
display(df.take(5))

df.columns

df.printSchema()

# for row in df.head(5):
#     print(row)
#     print('\n')

df.describe().show()

df.describe().printSchema()

from pyspark.sql.functions import format_number

result = df.describe()
result.select(result['province']
,format_number(result['specific'].cast('float'),2).alias('specific')
,format_number(result['general'].cast('float'),2).alias('general')
,format_number(result['year'].cast('int'),2).alias('year'),format_number(result['gdp'].cast('float'),2).alias('gdp')
,format_number(result['rnr'].cast('int'),2).alias('rnr'),format_number(result['rr'].cast('float'),2).alias('rr')
,format_number(result['fdi'].cast('int'),2).alias('fdi'),format_number(result['it'].cast('float'),2).alias('it')
,result['reg'].cast('string').alias('reg')
             ).show()

df2 = df.withColumn("specific_gdp_ratio",df["specific"]/(df["gdp"]*100))#.show()

df2.select('specific_gdp_ratio').show()

df.orderBy(df["specific"].asc()).head(1)[0][0]

from pyspark.sql.functions import mean
df.select(mean("specific")).show()

from pyspark.sql.functions import max,min

df.select(max("specific"),min("specific")).show()

df.filter("specific < 60000").count()

df.filter(df['specific'] < 60000).count()

from pyspark.sql.functions import count
result = df.filter(df['specific'] < 60000)
result.select(count('specific')).show()

(df.filter(df["gdp"]>8000).count()*1.0/df.count())*100

from pyspark.sql.functions import corr
df.select(corr("gdp","fdi")).show()

from pyspark.sql.functions import year
#yeardf = df.withColumn("Year",year(df["year"]))

max_df = df.groupBy('year').max()

max_df.select('year','max(gdp)').show()


from pyspark.sql.functions import month

#df.select("year","avg(gdp)").orderBy('year').show()
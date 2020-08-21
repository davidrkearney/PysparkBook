# Handling Missing Data with Pyspark

# Load data from a CSV
file_location = "/FileStore/tables/df_panel_fix.csv"
df = spark.read.format("CSV").option("inferSchema", True).option("header", True).load(file_location)
display(df.take(5))

df.show()

# Has to have at least 2 NON-null values
df.na.drop(thresh=2).show()

# Drop any row that contains missing data
df.na.drop().show()

df.na.drop(subset=["general"]).show()

df.na.drop(how='any').show()

df.na.drop(how='all').show()

df.na.fill('example').show()

df.na.fill(0).show()

df.na.fill('example',subset=['fr']).show()

df.na.fill(0,subset=['general']).show()

# Mean Imputation
from pyspark.sql.functions import mean
mean_val = df.select(mean(df['general'])).collect()

mean_val[0][0]

mean_gen = mean_val[0][0]

df.na.fill(mean_gen,["general"]).show()

df.na.fill(df.select(mean(df['general'])).collect()[0][0],['general']).show()
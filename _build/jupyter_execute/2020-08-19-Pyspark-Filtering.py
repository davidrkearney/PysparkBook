# Dataframe Filitering and Operations with Pyspark

# Load data from a CSV
file_location = "/FileStore/tables/df_panel_fix.csv"
df = spark.read.format("CSV").option("inferSchema", True).option("header", True).load(file_location)
display(df.take(5))

df.filter("specific<10000").show()

df.filter("specific<10000").select('province').show()


df.filter("specific<10000").select(['province','year']).show()

df.filter(df["specific"] < 10000).show()

df.filter((df["specific"] < 55000) & (df['gdp'] > 200) ).show()

df.filter((df["specific"] < 55000) | (df['gdp'] > 20000) ).show()

df.filter((df["specific"] < 55000) & ~(df['gdp'] > 20000) ).show()

df.filter(df["specific"] == 8964.0).show()


df.filter(df["province"] == "Zhejiang").show()


df.filter(df["specific"] == 8964.0).collect()


result = df.filter(df["specific"] == 8964.0).collect()


type(result[0])

row = result[0]

row.asDict()

for item in result[0]:
    print(item)
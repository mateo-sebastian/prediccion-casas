from sklearn.datasets import fetch_california_housing

datos = fetch_california_housing(as_frame=True)
df = datos.frame
df.to_csv("data/california_housing.csv", index=False)

print(df.shape)
print(df.head()) 
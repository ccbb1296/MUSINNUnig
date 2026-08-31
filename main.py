from sklearn.linear_model import LinearRegression

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
x = data.data
y = data.target

# 데이터 프레임 만들기
df = pd.DataFrame(x, columns=data.feature_names)

# 주택 가격 추가
df["MedHouseVal"] = y
print(df.head())

# csv 저장
df.to_csv("california_housing.csv", index=False)
print(df.dropna())
print(df.describe())
print(df.isnull().sum())
print(df.shape)


# 가격 분포 확인 (hist)
plt.figure(figsize=(5, 5))
sns.histplot(df["MedHouseVal"], bins=30)

plt.title("Distribution of House Prices")
plt.xlabel("House Price")
plt.ylabel("Frequency")

plt.show()

# x y 설정
x = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, random_state=42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 모델
m = LinearRegression()

# 학습
m.fit(x_train, y_train)

# 예측
y_pred = m.predict(x_test)

# 평가
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

# 출력
print("r2 : ", r2)
print("mae : ", mae)
print("rmse : ", rmse)

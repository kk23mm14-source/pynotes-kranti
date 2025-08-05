---
title: Sample
date: 2025-08-05
author: Your Name
cell_count: 21
score: 20
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
```


```python
diabetes = load_diabetes()
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
```


```python
df_orig = df.copy()

```


```python
df.loc[0:5, 'bmi'] = np.nan
df.loc[10:12, 'bp'] = np.nan
print("🔹 Missing values before filling:\n", df.isnull().sum())
```

    🔹 Missing values before filling:
     age       0
    sex       0
    bmi       6
    bp        3
    s1        0
    s2        0
    s3        0
    s4        0
    s5        0
    s6        0
    target    0
    dtype: int64
    


```python
df.fillna(df.median(numeric_only=True), inplace=True)
print("🔹 After imputation:\n", df.isnull().sum())

```

    🔹 After imputation:
     age       0
    sex       0
    bmi       0
    bp        0
    s1        0
    s2        0
    s3        0
    s4        0
    s5        0
    s6        0
    target    0
    dtype: int64
    


```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
```


```python
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

df_minmax = df.copy()
df_standard = df.copy()

df_minmax[diabetes.feature_names] = scaler_minmax.fit_transform(df[diabetes.feature_names])
df_standard[diabetes.feature_names] = scaler_standard.fit_transform(df[diabetes.feature_names])
```


```python
df['bmi_bin'] = pd.cut(df['bmi'], bins=3, labels=["Low", "Medium", "High"])

```


```python
df['age_bmi'] = df['age'] * df['bmi']
df['bp_hdl'] = df['bp'] * df['s5']
```


```python
X = df[diabetes.feature_names]
y = df['target']
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print("🔹 Top 5 features selected:", selected_features.tolist())
```

    🔹 Top 5 features selected: ['bmi', 'bp', 's3', 's4', 's5']
    


```python
df[diabetes.feature_names].hist(figsize=(12, 8), bins=15, edgecolor='black')
plt.suptitle("Histograms of Features")
plt.tight_layout()
plt.show()

```


    
![png](/pynotes-kranti/images/sample_10_0.png)
    



```python
df[diabetes.feature_names].plot(kind='box', subplots=True, layout=(3, 4), figsize=(14, 10), title="Boxplots")
plt.tight_layout()
plt.show()
```


    
![png](/pynotes-kranti/images/sample_11_0.png)
    



```python
means = df[diabetes.feature_names].mean()
means.plot(kind='line', marker='o')
plt.title("Mean of Each Feature")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](/pynotes-kranti/images/sample_12_0.png)
    



```python
corr = df.select_dtypes(include='number').corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
```


    
![png](/pynotes-kranti/images/sample_13_0.png)
    



```python
from pandas.plotting import scatter_matrix
scatter_matrix(df[selected_features.tolist() + ['target']], figsize=(12, 10), alpha=0.8, diagonal='hist')
plt.suptitle("Scatter Matrix (Top 5 Features + Target)")
plt.show()
```


    
![png](/pynotes-kranti/images/sample_14_0.png)
    



```python
plt.figure(figsize=(6, 6))
df['bmi_bin'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title("BMI Binning Distribution")
plt.ylabel("")
plt.axis('equal')
plt.show()
```


    
![png](/pynotes-kranti/images/sample_15_0.png)
    



```python
mean_grouped = df.groupby('bmi_bin', observed=True)[selected_features].mean()
mean_grouped.plot(kind='line', marker='o')
plt.title("Mean of Top Features by BMI Group")
plt.ylabel("Mean Value")
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](/pynotes-kranti/images/sample_16_0.png)
    



```python
for feature in selected_features:
    plt.figure(figsize=(6, 4))
    for i, group in enumerate(df['bmi_bin'].dropna().unique()):
        data = df[df['bmi_bin'] == group][feature]
        plt.boxplot(data, positions=[i])
    plt.xticks(range(len(df['bmi_bin'].dropna().unique())), df['bmi_bin'].dropna().unique())
    plt.title(f"Custom Violin-like Plot for {feature}")
    plt.ylabel(feature)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
```


    
![png](/pynotes-kranti/images/sample_17_0.png)
    



    
![png](/pynotes-kranti/images/sample_17_1.png)
    



    
![png](/pynotes-kranti/images/sample_17_2.png)
    



    
![png](/pynotes-kranti/images/sample_17_3.png)
    



    
![png](/pynotes-kranti/images/sample_17_4.png)
    



```python
# Skewness and Kurtosis
skewness = df[diabetes.feature_names].skew()
kurtosis = df[diabetes.feature_names].kurt()

print("🔹 Skewness:\n", skewness)
print("🔹 Kurtosis:\n", kurtosis)

```

    🔹 Skewness:
     age   -0.255916
    sex    0.113063
    bmi    0.454400
    bp     0.308706
    s1     0.126606
    s2     0.093401
    s3     0.419665
    s4     0.671554
    s5     0.236942
    s6     0.143236
    dtype: float64
    🔹 Kurtosis:
     age   -0.674778
    sex   -1.997006
    bmi   -0.305152
    bp    -0.478878
    s1    -0.232286
    s2    -0.213673
    s3    -0.254227
    s4     0.249596
    s5    -0.190126
    s6    -0.228353
    dtype: float64
    


```python
df_log = df.copy()
for col in diabetes.feature_names:
    df_log[col] = np.log1p(df_log[col] - df_log[col].min() + 1)  # to handle negative values

```


```python

```


---
**Score: 20**
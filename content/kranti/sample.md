---
title: Sample
date: 2025-08-05
author: Your Name
cell_count: 38
score: 35
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
    bmi    0.462780
    bp     0.312219
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
    bmi   -0.262762
    bp    -0.459589
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
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=y, cmap='viridis')
plt.title("PCA Components Scatter Plot")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Target')
plt.grid(True)
plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print("🔹 Polynomial Features Shape:", X_poly.shape)

```


    
![png](/pynotes-kranti/images/sample_20_0.png)
    


    🔹 Polynomial Features Shape: (409, 65)
    


```python
df['s1_s2_interaction'] = df['s1'] * df['s2']
df['log_age'] = np.log1p(df['age'] - df['age'].min() + 1)
le = LabelEncoder()
df['bmi_bin_encoded'] = le.fit_transform(df['bmi_bin'])
print("🔹 BMI bin encoding:\n", df[['bmi_bin', 'bmi_bin_encoded']].drop_duplicates())

```

    🔹 BMI bin encoding:
       bmi_bin  bmi_bin_encoded
    0  Medium                2
    6     Low                1
    8    High                0
    


```python
import seaborn as sns
sns.kdeplot(df['target'], fill=True)
plt.title("KDE Plot of Target Variable")
plt.grid(True)
plt.show()

```


    
![png](/pynotes-kranti/images/sample_22_0.png)
    



```python
pip install seaborn

```

    Requirement already satisfied: seaborn in c:\users\krant\miniconda3\envs\py12\lib\site-packages (0.13.2)
    Requirement already satisfied: numpy!=1.24.0,>=1.20 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from seaborn) (2.3.2)
    Requirement already satisfied: pandas>=1.2 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from seaborn) (2.3.1)
    Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from seaborn) (3.10.5)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.3)
    Requirement already satisfied: cycler>=0.10 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.59.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (25.0)
    Requirement already satisfied: pillow>=8 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (11.3.0)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from pandas>=1.2->seaborn) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from pandas>=1.2->seaborn) (2025.2)
    Requirement already satisfied: six>=1.5 in c:\users\krant\miniconda3\envs\py12\lib\site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.17.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
sorted_target = np.sort(df['target'])
cdf = np.arange(len(sorted_target)) / float(len(sorted_target))
plt.plot(sorted_target, cdf)
plt.title("CDF of Target")
plt.xlabel("Target")
plt.ylabel("CDF")
plt.grid(True)
plt.show()

```


    
![png](/pynotes-kranti/images/sample_24_0.png)
    



```python
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
df_robust = df.copy()
df_robust[diabetes.feature_names] = robust_scaler.fit_transform(df[diabetes.feature_names])

```


```python
corr_original = df[diabetes.feature_names].corr()
corr_scaled = df_minmax[diabetes.feature_names].corr()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(corr_original, cmap='coolwarm')
plt.title("Original Features Correlation")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(corr_scaled, cmap='coolwarm')
plt.title("MinMax Scaled Features Correlation")
plt.colorbar()
plt.tight_layout()
plt.show()

```


    
![png](/pynotes-kranti/images/sample_26_0.png)
    



```python

```


```python
df_missing = df.copy()
np.random.seed(42)
for col in ['s1', 's2', 's3']:
    df_missing.loc[df_missing.sample(frac=0.1).index, col] = np.nan
```


```python
df_missing.fillna(df_missing.select_dtypes(include='number').median(), inplace=True)
df_missing['bmi_age'] = df_missing['bmi'] * df_missing['age']
df_missing['s1_log'] = np.log1p(df_missing['s1'] - df_missing['s1'].min() + 1)
```


```python
for col in diabetes.feature_names:
    print(f"{col} - Skew: {df_missing[col].skew():.2f}, Kurtosis: {df_missing[col].kurtosis():.2f}")

```

    age - Skew: -0.26, Kurtosis: -0.67
    sex - Skew: 0.11, Kurtosis: -2.00
    bmi - Skew: 0.46, Kurtosis: -0.26
    bp - Skew: 0.31, Kurtosis: -0.46
    s1 - Skew: 0.16, Kurtosis: 0.13
    s2 - Skew: 0.07, Kurtosis: 0.09
    s3 - Skew: 0.47, Kurtosis: 0.03
    s4 - Skew: 0.67, Kurtosis: 0.25
    s5 - Skew: 0.24, Kurtosis: -0.19
    s6 - Skew: 0.14, Kurtosis: -0.23
    


```python
df_missing['bmi_category'] = pd.cut(df_missing['bmi'], bins=3, labels=["Low", "Medium", "High"])
df_missing['bmi_encoded'] = LabelEncoder().fit_transform(df_missing['bmi_category'])
```


```python
for col in ['bmi', 's1', 's2']:
    plt.figure()
    df_missing.boxplot(column=col)
    plt.title(f'Boxplot of {col}')
    plt.grid(True)
    plt.show()

```


    
![png](/pynotes-kranti/images/sample_32_0.png)
    



    
![png](/pynotes-kranti/images/sample_32_1.png)
    



    
![png](/pynotes-kranti/images/sample_32_2.png)
    



```python
for col in ['bmi', 'bp', 's5']:
    plt.figure()
    sns.kdeplot(df_missing[col], fill=True)
    plt.title(f'Distribution of {col}')
    plt.grid(True)
    plt.show()

```


    
![png](/pynotes-kranti/images/sample_33_0.png)
    



    
![png](/pynotes-kranti/images/sample_33_1.png)
    



    
![png](/pynotes-kranti/images/sample_33_2.png)
    



```python
import seaborn as sns
sns.pairplot(df_missing[['bmi', 'bp', 's5', 'target']], diag_kind='kde')
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()

```


    
![png](/pynotes-kranti/images/sample_34_0.png)
    



```python
plt.figure(figsize=(12, 8))
cor = df_missing.select_dtypes(include='number').corr()
plt.imshow(cor, cmap='coolwarm')
plt.title("Correlation Matrix (with Engineered Features)")
plt.colorbar()
plt.xticks(ticks=np.arange(len(cor.columns)), labels=cor.columns, rotation=90)
plt.yticks(ticks=np.arange(len(cor.columns)), labels=cor.columns)
plt.tight_layout()
plt.show()

```


    
![png](/pynotes-kranti/images/sample_35_0.png)
    



```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

X_scaled = StandardScaler().fit_transform(df_missing[diabetes.feature_names])
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(pca_data)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
plt.title("PCA with KMeans Clusters")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

```


    
![png](/pynotes-kranti/images/sample_36_0.png)
    



```python

```


---
**Score: 35**
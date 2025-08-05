---
title: Assaignment
date: 2025-08-05
author: Your Name
cell_count: 10
score: 10
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
```


```python
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df_cleaned = df.dropna(axis=1, how='all').copy()

```


```python
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Summary Stats ---")
print(df.describe())
print("\n--- Class Distribution ---")
print(df['target'].value_counts())
```

    
    --- Dataset Info ---
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 14 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   alcohol                       178 non-null    float64
     1   malic_acid                    178 non-null    float64
     2   ash                           178 non-null    float64
     3   alcalinity_of_ash             178 non-null    float64
     4   magnesium                     178 non-null    float64
     5   total_phenols                 178 non-null    float64
     6   flavanoids                    178 non-null    float64
     7   nonflavanoid_phenols          178 non-null    float64
     8   proanthocyanins               178 non-null    float64
     9   color_intensity               178 non-null    float64
     10  hue                           178 non-null    float64
     11  od280/od315_of_diluted_wines  178 non-null    float64
     12  proline                       178 non-null    float64
     13  target                        178 non-null    int64  
    dtypes: float64(13), int64(1)
    memory usage: 19.6 KB
    None
    
    --- Summary Stats ---
              alcohol  malic_acid         ash  alcalinity_of_ash   magnesium  \
    count  178.000000  178.000000  178.000000         178.000000  178.000000   
    mean    13.000618    2.336348    2.366517          19.494944   99.741573   
    std      0.811827    1.117146    0.274344           3.339564   14.282484   
    min     11.030000    0.740000    1.360000          10.600000   70.000000   
    25%     12.362500    1.602500    2.210000          17.200000   88.000000   
    50%     13.050000    1.865000    2.360000          19.500000   98.000000   
    75%     13.677500    3.082500    2.557500          21.500000  107.000000   
    max     14.830000    5.800000    3.230000          30.000000  162.000000   
    
           total_phenols  flavanoids  nonflavanoid_phenols  proanthocyanins  \
    count     178.000000  178.000000            178.000000       178.000000   
    mean        2.295112    2.029270              0.361854         1.590899   
    std         0.625851    0.998859              0.124453         0.572359   
    min         0.980000    0.340000              0.130000         0.410000   
    25%         1.742500    1.205000              0.270000         1.250000   
    50%         2.355000    2.135000              0.340000         1.555000   
    75%         2.800000    2.875000              0.437500         1.950000   
    max         3.880000    5.080000              0.660000         3.580000   
    
           color_intensity         hue  od280/od315_of_diluted_wines      proline  \
    count       178.000000  178.000000                    178.000000   178.000000   
    mean          5.058090    0.957449                      2.611685   746.893258   
    std           2.318286    0.228572                      0.709990   314.907474   
    min           1.280000    0.480000                      1.270000   278.000000   
    25%           3.220000    0.782500                      1.937500   500.500000   
    50%           4.690000    0.965000                      2.780000   673.500000   
    75%           6.200000    1.120000                      3.170000   985.000000   
    max          13.000000    1.710000                      4.000000  1680.000000   
    
               target  
    count  178.000000  
    mean     0.938202  
    std      0.775035  
    min      0.000000  
    25%      0.000000  
    50%      1.000000  
    75%      2.000000  
    max      2.000000  
    
    --- Class Distribution ---
    target
    1    71
    0    59
    2    48
    Name: count, dtype: int64
    


```python
df.loc[0:5, 'bmi'] = np.nan
df.loc[10:12, 'bp'] = np.nan
print("🔹 Missing values before filling:\n", df.isnull().sum())
```

    🔹 Missing values before filling:
     alcohol                           0
    malic_acid                        0
    ash                               0
    alcalinity_of_ash                 0
    magnesium                         0
    total_phenols                     0
    flavanoids                        0
    nonflavanoid_phenols              0
    proanthocyanins                   0
    color_intensity                   0
    hue                               0
    od280/od315_of_diluted_wines      0
    proline                           0
    target                            0
    bmi                             178
    bp                              178
    dtype: int64
    


```python
np.random.seed(0)
for col in df.columns[:-1]:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

print("\n--- Missing After Injection ---")
print(df.isnull().sum())
```

    
    --- Missing After Injection ---
    alcohol                           9
    malic_acid                        9
    ash                               9
    alcalinity_of_ash                 9
    magnesium                         9
    total_phenols                     9
    flavanoids                        9
    nonflavanoid_phenols              9
    proanthocyanins                   9
    color_intensity                   9
    hue                               9
    od280/od315_of_diluted_wines      9
    proline                           9
    target                            9
    bmi                             178
    bp                              178
    dtype: int64
    


```python
from sklearn.impute import SimpleImputer
df_cleaned = df.dropna(axis=1, how='all').copy()
features = df_cleaned.columns[:-1]
imputer = SimpleImputer(strategy='mean')
df_cleaned.loc[:, features] = imputer.fit_transform(df_cleaned[features])

```


```python

df_cleaned = df.dropna(axis=1, how='all').copy()
features = df_cleaned.columns[:-1]  
imputer = SimpleImputer(strategy='mean')
df_cleaned.loc[:, features] = imputer.fit_transform(df_cleaned[features])
scaler = StandardScaler()
df_cleaned.loc[:, features] = scaler.fit_transform(df_cleaned[features])

```


```python
plt.figure(figsize=(10, 6))
plt.matshow(df.corr(), fignum=1)
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.title("Correlation Matrix", pad=100)
plt.show()
```


    
![png](/pynotes-kranti/images/assaignment_7_0.png)
    



```python

for col in df.columns[:-1]:
    if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
        plt.figure()
        plt.hist(df[col].dropna(), bins=20)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
    else:
        print(f"Skipping column: {col} (non-numeric or all NaN)")


```


    
![png](/pynotes-kranti/images/assaignment_8_0.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_1.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_2.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_3.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_4.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_5.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_6.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_7.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_8.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_9.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_10.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_11.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_12.png)
    



    
![png](/pynotes-kranti/images/assaignment_8_13.png)
    


    Skipping column: bmi (non-numeric or all NaN)
    


```python

```


---
**Score: 10**
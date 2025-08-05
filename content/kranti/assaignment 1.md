---
title: Assaignment 1
date: 2025-08-05
author: Your Name
cell_count: 5
score: 5
---

```python
import pandas as pd
import numpy as np
```


```python
def avg(l):
  sum=0
  count=0
  for e in l:
    sum=sum+e
    count=count+1
  avg=sum/count
  print(avg)
```


```python
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
avg(speed)
```

    89.76923076923077
    


```python
import numpy
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = numpy.mean(speed)
print(x)
```

    89.76923076923077
    


```python

```


---
**Score: 5**
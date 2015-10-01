# py-soft-impute
Python implementation of Mazumder and Hastie's R softImpute package.

This code provides an experimental sklearn-ish class for missing data imputation. The code is currently
more or less a literal translation from the original R package's simpute.als function. 
I'm planning on also implementing simpute.svd.
Hastie and Mazumder experiment with this approach on the Netflix problem.

Notes:
- Missing values are represented by nan
- For additional detail the R vignette [here](https://web.stanford.edu/~hastie/swData/softImpute/vignette.html) is quite helpful
- R package: https://github.com/cran/softImpute
- paper: http://arxiv.org/abs/1410.2596


### Toy example usage
```python
 import numpy as np
 from soft_impute import SoftImpute

 X = np.arange(50).reshape(10, 5) * 1.0

 # Change 10 to nan aka missing
 X[2, 0] = np.nan
 clf.fit(X)
 imputed = clf.transform(X)

 # Should be 10
 print imputed[2, 0]
 array([[  0.        ,   1.        ,   2.        ,   3.        ,   4.        ],
       [  5.        ,   6.        ,   7.        ,   8.        ,   9.        ],
       [ 10.01385116,  11.        ,  12.        ,  13.        ,  14.        ],
       [ 15.        ,  16.        ,  17.        ,  18.        ,  19.        ],
       [ 20.        ,  21.        ,  22.        ,  23.        ,  24.        ],
       [ 25.        ,  26.        ,  27.        ,  28.        ,  29.        ],
       [ 30.        ,  31.        ,  32.        ,  33.        ,  34.        ],
       [ 35.        ,  36.        ,  37.        ,  38.        ,  39.        ],
       [ 40.        ,  41.        ,  42.        ,  43.        ,  44.        ],
       [ 45.        ,  46.        ,  47.        ,  48.        ,  49.        ]])
 ```

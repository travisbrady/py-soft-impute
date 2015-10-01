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
 10.01385116
 ```

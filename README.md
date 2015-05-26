# py-soft-impute
Python implementation of Mazumder and Hastie's R softImpute package.

This class provides an experimental class for missing data imputation. The code is currently
more or less a literal translation from the original R package's simpute.als function. 
I'm planning on also implementing simpute.svd.
Hastie and Mazumder experiment with this approach on the Netflix problem.

- R package: https://github.com/cran/softImpute
- paper: http://arxiv.org/abs/1410.2596

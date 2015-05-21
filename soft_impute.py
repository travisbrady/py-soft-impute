import numpy as np


def frob(Uold, Dsqold, Vold, U, Dsq, V):
    denom = (Dsqold ** 2).sum()
    utu = Dsq * (U.T.dot(Uold))
    vtv = Dsqold * (Vold.T.dot(V))
    uvprod = utu.dot(vtv).diagonal().sum()
    num = denom + (Dsqold ** 2).sum() - 2*uvprod
    return num / max(denom, 1e-9)

class SoftImpute:
    def __init__(self):
        self.u = None
        self.d = None
        self.v = None

    def fit(self, X, J=2, thresh=1e-05, _lambda=0, maxit=100):
        n,m = X.shape
        xnas = np.isnan(X)
        nz = m*n - xnas.sum()
        xfill = X.copy()
        V = np.zeros((m, J))
        U = np.random.normal(0.0, 1.0, (n, J))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        #Dsq = np.ones((1, J))
        Dsq = np.ones((J, 1))
        xfill[xnas] = 0.0
        ratio = 1.0
        while ratio > thresh and maxit > 0:
            maxit -= 1
            U_old = U
            V_old = V
            Dsq_old = Dsq
            B = U.T.dot(xfill)

            if _lambda > 0:
                tmp = (Dsq / (Dsq + _lambda))
                B = B * tmp

            Bsvd = np.linalg.svd(B.T, full_matrices=False)
            V = Bsvd[0]
            Dsq = Bsvd[1][:, np.newaxis]
            U = U.dot(Bsvd[2])

            tmp = Dsq * V.T

            xhat = U.dot(tmp)

            xfill[xnas] = xhat[xnas]
            A = xfill.dot(V).T
            Asvd = np.linalg.svd(A.T, full_matrices=False)
            U = Asvd[0]
            Dsq = Asvd[1][:, np.newaxis]
            V = V.dot(Asvd[2])
            tmp = Dsq * V.T

            xhat = U.dot(tmp)
            xfill[xnas] = xhat[xnas]
            ratio = frob(U_old, Dsq_old, V_old, U, Dsq, V)
            print 'ratio = {:.5f} maxit = {}'.format(ratio, maxit)

        self.u = U[:,:J]
        self.d = Dsq[:J]
        self.v = V[:,:J]
        self.lambda_ = _lambda
        return self

    def suv(self, X, vd):
        X_imp = X.copy()
        res = self.u.dot(vd.T)
        np.copyto(X_imp, res, where=np.isnan(X_imp))
        '''
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.isnan(X[i, j]):
                    #yex = self.u[i,:].dot(vd.T[j,:])
                    #yex = self.u[i,:] * vd[:,j]
                    yex = self.u.dot(vd.T)[i, j]
                    X_imp[i, j] = yex
        '''
        return X_imp

    def predict(self, X):
        vd = self.v * np.outer(np.ones(self.v.shape[0]), self.d)
        out = self.suv(X, vd)
        return out

def main():
    X = np.random.random((10,3)) + (np.arange(10).reshape(10,1) ** 2)
    #X = np.arange(100).reshape(25, 4).astype(np.float32)

    clf = SoftImpute()
    fit = clf.fit(X, J=2, _lambda=0.0)
    print fit.u
    print fit.d
    print fit.v
    X_test = X.copy()
    X_test[3,1] = np.nan
    X_imp = clf.predict(X_test)
    print 'XXX'
    print X_test[:5]
    print
    print 'X_imp'
    print X_imp[:5]

if __name__ == '__main__':
    main()


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
        xfill = X
        V = np.zeros((m, J))
        U = np.random.normal(0.0, 1.0, (n, J))
        U, _, _ = np.linalg.svd(U, full_matrices=False)
        Dsq = np.ones((1, J))
        xfill[xnas] = 0.0
        ratio = 1.0
        while ratio > thresh and maxit > 0:
            print 'maxit = {}'.format(maxit)
            maxit -= 1
            U_old = U
            V_old = V
            Dsq_old = Dsq
            B = U.T.dot(xfill)

            if _lambda > 0:
                tmp = (Dsq / (Dsq + _lambda)).T
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
            ratio =frob(U_old, Dsq_old, V_old, U, Dsq, V)
        self.u = U[:,:J]
        self.d = Dsq[:J]
        self.v = V[:,:J]
        self.lambda_ = _lambda
        return self

    def suv(self, vd):
        r = []
        print vd.shape
        print self.u.shape
        return self.u.T.dot(vd)

    def predict(self, X):
        vd = np.outer(np.ones(self.v.shape[0]), self.d)
        out = self.suv(vd)
        return out

def main():
    X = np.random.random((10,3)) + np.arange(10).reshape(10,1)
    #outs = 'matrix(' + str(b.tolist()).replace('[', 'c(').replace(']', ')')
    #outs += ',nrow=%d, ncol=%d)' % (b.shape[0], b.shape[1])

    clf = SoftImpute()
    fit = clf.fit(X, _lambda=0.0)
    print
    print fit.u
    print fit.d
    print fit.v
    print
    print clf.predict(X)

if __name__ == '__main__':
    main()


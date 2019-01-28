import numpy as np


class SYSREM:
    def __init__(self, iteration_cap=250, pre_processing=True, norm_method='normalize', after_processing=False, multiprocess=False):
        self._iter_cap = iteration_cap
        self._pre_processing = pre_processing
        self._norm_method = norm_method
        self._after_processing = after_processing
        self._multiprocess = multiprocess
        self._comp = None
        self._data = None
        self._n_rows = None
        self._n_col = None
        self._sigma_map = None
        self._a = None
        self._c = None
        self._S = None
        self._threshold = 10. ** (-5.)
        self.components_ = {}
        self.a_ = {}
        self.c_ = {}
        self.S_squared_ = []

    def __matx_prod(self):
        c = np.empty((self._n_rows, self._n_col))
        for i in range(0, self._n_rows):
            for j in range(0, self._n_col):
                c[i, j] = np.sum(self._c[i, :] * self._a[:, j])
        return c

    def __sig_map(self):
        for i in range(0, self._n_rows):
            for j in range(0, self._n_col):
                self._sigma_map[i, j] = np.sqrt((np.std(self._data[i, :]) ** 2.) + (np.std(self._data[:, j]) ** 2.))

    def __c_coeff(self):
        for i in range(0, self._n_rows):
            c_term_num = np.sum((self._data[i, :] * self._a[0, :]) / (self._sigma_map[i, :] ** 2.))
            c_term_den = np.sum((self._a[0, :] ** 2.) / (self._sigma_map[i, :] ** 2.))
            self._c[i, 0] = c_term_num / c_term_den

    def __a_coeff(self):
        for j in range(0, self._n_col):
            a_term_num = np.sum((self._data[:, j] * self._c[:, 0]) / (self._sigma_map[:, j] ** 2.))
            a_term_den = np.sum((self._c[:, 0] ** 2.) / (self._sigma_map[:, j] ** 2.))
            self._a[0, j] = a_term_num / a_term_den

    def __S_func(self):
        if self._multiprocess:
            S_term_num = (self._data - self.__matx_prod()) ** 2.
        else:
            S_term_num = (self._data - np.dot(self._c, self._a)) ** 2.
        S_term_den = self._sigma_map ** 2.
        self._S = S_term_num / S_term_den
        self._S = np.sum(np.sum(self._S, axis=0))
        return self._S

    def __pre_processing(self):
        if self._norm_method == 'normalize':
            norm = PRE_PROC(self._data)
            self._data = norm.normaliz()
        elif self._norm_method == 'standardize':
            stsc = PRE_PROC(self._data)
            self._data = stsc.standardiz()
        self._data = self._data.T

    def components_(self):
        if len(self.components_.keys()) == 0:
            raise RuntimeError('Components have not been calculated yet. Try to fit the class to your data and transform it.')
        else:
            return self.components_

    def a_(self):
        if len(self.a_.keys()) == 0:
            raise RuntimeError('Components have not been calculated yet. Try to fit the class to your data and transform it.')
        else:
            return self.a_

    def c_(self):
        if len(self.c_.keys()) == 0:
            raise RuntimeError('Components have not been calculated yet. Try to fit the class to your data and transform it.')
        else:
            return self.c_

    def fit(self, matrx):
        if len(matrx.shape) != 2:
            raise IOError('SYSREM input must be 2D array')

        self._a = np.array([np.random.random(matrx.shape[0])])
        self._c = np.array([np.random.random(matrx.shape[1])]).T
        self._data = np.array(matrx + 0.0)
        self._n_rows = self._data.shape[0]
        self._n_col = self._data.shape[1]

        if self._pre_processing:
            self.__pre_processing()
        else:
            self._data = self._data.T

        self._n_rows = self._data.shape[0]
        self._n_col = self._data.shape[1]
        self._sigma_map = np.empty((self._n_rows, self._n_col))
        self.__sig_map()

    def transform(self, components=10):
        if self._data is None:
            raise RuntimeError('Object needs to be fit to the data before calling transform()')

        self._comp = components
        for comp in range(0, self._comp):
            S = [np.random.random()]
            S_pred = 0.
            loop = 0
            while np.abs(S_pred - S[-1]) > self._threshold:
                S_pred = S[-1]
                self.__a_coeff()
                self.__c_coeff()
                S.append(float(self.__S_func()))
                if loop == self._iter_cap:
                    break
                else:
                    loop += 1

            self.S_squared_.append(S[-1])

            if self._multiprocess:
                self.components_[str(comp + 1)] = self.__matx_prod().T
                self._data -= self.__matx_prod()
            else:
                self.components_[str(comp + 1)] = np.dot(self._c, self._a).T
                self._data -= np.dot(self._c, self._a)

            self.a_[str(comp + 1)] = self._a
            self.c_[str(comp + 1)] = self._c

        self._data = self._data.T
        if self._after_processing:
            for j in range(0, self._n_rows):
                self._data[:, j] /= np.std(self._data[:, j])

        return self._data

    def fit_transform(self, matrx, components=10):
        self.fit(matrx)
        result = self.transform(components)
        return result


class PCA:
    def __init__(self, pre_processing=True, norm_method='normalize', after_processing=False, multiprocess=False):
        self._pre_processing = pre_processing
        self._norm_method = norm_method
        self._after_processing = after_processing
        self._multiprocess = multiprocess
        self._data = None
        self._n_rows = None
        self._n_col = None
        self._cov_mat = None
        self._eival = None
        self._eivet = None
        self._c = None
        self.variance_explained_ = None
        self.components_ = {}

    def __matx_prod(self, m1, m2):
        c = np.empty((self._n_rows, self._n_col))
        for i in range(0, self._n_rows):
            for j in range(0, self._n_col):
                c[i, j] = np.sum(m1[i, :] * m2[:, j])
        return c

    def __pre_processing(self):
        if self._norm_method == 'normalize':
            norm = PRE_PROC(self._data)
            self._data = norm.normaliz()
        elif self._norm_method == 'standardize':
            stsc = PRE_PROC(self._data)
            self._data = stsc.standardiz()

    def fit(self, matrx):
        if len(matrx.shape) != 2:
            raise IOError('PCA input must be 2D array')

        self._data = np.array(matrx + 0.0)
        self._n_rows = self._data.shape[0]
        self._n_col = self._data.shape[1]

        if self._pre_processing:
            self.__pre_processing()

        self._cov_mat = np.cov(self._data)

        self._eival, self._eivet = np.linalg.eig(self._cov_mat)
        self._eival, self._eivet = np.real(self._eival), np.real(self._eivet)

        indx = self._eival.argsort()[::-1]
        self._eival = self._eival[indx]
        self._eivet = self._eivet[:, indx]

        # tot = np.sum(self._eival)
        # var_exp = [(i / tot) for i in self._eival]
        # cum_var_exp = np.cumsum(var_exp)

        # plt.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.5, align='center', label='individual explained variance')
        # plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='cumulative explained variance')
        # plt.grid()
        # plt.ylabel('Explained variance ratio')
        # plt.xlabel('Principal component')
        # plt.legend(loc='best')

        tot = np.sum(self._eival)
        self.variance_explained_ = [(i / tot) for i in self._eival]

        if self._multiprocess:
            self._c = self.__matx_prod(self._data.T, self._eivet)
        else:
            self._c = np.dot(self._data.T, self._eivet)

        for i in range(0, self._n_rows):
            if self._multiprocess:
                self.components_['comp_' + str(i + 1)] = (self.__matx_prod(self._c[:, i:i + 1], (self._eivet[:, i:i + 1]).T)).T
            else:
                self.components_['comp_' + str(i + 1)] = (np.dot(self._c[:, i:i + 1], (self._eivet[:, i:i + 1]).T)).T

    def transform(self, components=1, comp_end=(-1)):
        if self._data is None:
            raise RuntimeError('Object need to be fit to the data before calling transform()')

        if self._multiprocess:
            self._data = (self.__matx_prod(self._c[:, components:comp_end], (self._eivet[:, components:comp_end]).T)).T
        else:
            self._data = (np.dot(self._c[:, components:comp_end], (self._eivet[:, components:comp_end]).T)).T

        if self._after_processing:
            for j in range(0, self._n_col):
                self._data[:, j] /= np.std(self._data[:, j])
        result = self._data

        return result

    def fit_transform(self, matrx, components=1, comp_end=(-1)):
        self.fit(matrx)
        result = self.transform(components, comp_end)
        return result


class PRE_PROC:
    def __init__(self, matrx):
        self.data = matrx + 0.0
        self.n_rows = len(matrx[:, 0])
        self.n_col = len(matrx[0, :])

    def normaliz(self):
        for i in range(0, self.n_rows):
            self.data[i, :] /= np.median(self.data[i, :])
        for j in range(0, self.n_col):
            self.data[:, j] -= np.mean(self.data[:, j])
        return self.data

    def standardiz(self):
        for i in range(0, self.n_rows):
            self.data[i, :] /= np.median(self.data[i, :])
        for j in range(0, self.n_col):
            self.data[:, j] -= np.mean(self.data[:, j])
            self.data[:, j] /= np.std(self.data[:, j])
        return self.data

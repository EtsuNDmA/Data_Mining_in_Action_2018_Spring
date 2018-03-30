# coding=utf-8
import numpy as np
from sklearn.base import BaseEstimator


LR_PARAMS_DICT = {
    'C': 10.,
    'random_state': 777,
    'iters': 1000,
    'batch_size': 1000,
    'step': 0.01,
    'penalty': 'l2',
}


class MyLogisticRegression(BaseEstimator):
    def __init__(self, C=1.0, random_state=None, iters=100,
                 batch_size=1000, step=0.1, penalty='l2'
                ):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step
        self.penalty = penalty
    
    @staticmethod
    def sigmoid(z):
        return 1.0 / (1 + np.exp(-z))
    
    # будем пользоваться этой функцией для подсчёта <w, x>
    def __predict(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.w0)

    # sklearn нужно, чтобы predict возвращал классы, поэтому оборачиваем наш __predict в это
    def predict(self, X):
        res = self.__predict(X)
        res[res >= 0.5] = 1
        res[res < 0.5] = 0
        return res
    
    # производная регуляризатора
    def der_reg(self):
        if self.penalty == 'l1':
            return self.C * np.sign(self.w)
        elif self.penalty == 'l2':
            return self.C *self.w

    # будем считать стохастический градиент не на одном элементе, а сразу на пачке (чтобы было эффективнее)
    def der_loss(self, x, y):
        # x.shape == (batch_size, features)
        # y.shape == (batch_size,)

        # считаем производную по каждой координате на каждом объекте
        # для масштаба возвращаем средний градиент по пачке
        d = self.__predict(x) - y
        ders_w = np.mean(x * d[:, np.newaxis], axis=0)
        der_w0 = np.mean(d)
        return ders_w, der_w0

    def fit(self, X_train, y_train):
        # RandomState для воспроизводитмости
        random_gen = np.random.RandomState(self.random_state)
        
        # получаем размерности матрицы
        size, dim = X_train.shape
        
        # случайная начальная инициализация
        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()
        self.learning_curve = np.zeros(self.iters)
        for i in range(self.iters):  
            # берём случайный набор элементов
            rand_indices = random_gen.choice(size, self.batch_size)
            # исходные метки классов это 0/1
            X = X_train[rand_indices]
            y = y_train[rand_indices]
            self.learning_curve[i] += log_loss(y,self.__predict(X), labels=(0,1))
            # считаем производные
            der_w, der_w0 = self.der_loss(X, y)
            der_w += self.der_reg()
            # обновляемся по антиградиенту
            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step
        # метод fit для sklearn должен возвращать self
        return self
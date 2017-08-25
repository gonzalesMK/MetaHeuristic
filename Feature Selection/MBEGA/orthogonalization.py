# -*- coding: utf-8 -*-
import numpy as np

v = np.diag(np.ones(6), 0)

d = np.asarray([1, 1, 1, 1, 1, 1])

a = np.array([ sum([ d[i] * v[i] for i in range (j,v.shape[1])])  for j in range(0,v.shape[1])])

w = np.zeros(a.shape)

for i in range(0,v.shape[1]):
    w[i] = a[i] - [sum([np.dot(a[i], v[j]) * v[j] for j in range(0,i)])]
    v[i]  = w[i]







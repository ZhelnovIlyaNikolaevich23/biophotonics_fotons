import random
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# количество фотонов
N = 1000

# внутренний и внешний радиусы пучка
a1 = 2
a2 = 6
a = np.log(a2 / a1)

#количество бинов
bin = 30

dr = (a2-a1)/bin


def dist():
    prob_density = []
    r = random.uniform(a1, a2)
    u = random.random()
    X = (1/(a*r)) * np.cos(2*np.pi * u)
    Y = (1/(a*r)) * np.sin(2*np.pi * u)
    R = (1/(a*r))
    prob_density.append(X)
    prob_density.append(Y)
    prob_density.append(R)
    return prob_density

list_X = []
list_Y = []
list_R = []

for i in range(N):
    rand = dist()
    list_X.append(rand[0])
    list_Y.append(rand[1])
    list_R.append(rand[2])

fg, ax = plt.subplots()
plt.scatter(list_X, list_Y, marker="o", color="g")
#plt.title()
plt.xlabel('x')
plt.ylabel('y')

plt.show()

#histogram
dr = (max(list_R)-min(list_R))/bin

num_R = {}
list_cel_r = []

for i in range(bin):
    num = 0
    for r in list_R:
        if (r >= min(list_R) + dr * i) and (r <= min(list_R) + dr * (i + 1)):
            num += 1
    num_R[i] = num

pdf_R = {}
for i in range(bin):
    S = np.pi * ((a1 + dr * (i + 1))**2 - (a1 + dr * i)**2)
    pdf_R[i] = num_R[i]/(N)



list_hist_x = []
for y in list_Y:
    if (y >= -0.1) and (y <= 0.1):
        list_hist_x.append(list_X[list_Y.index(y)])


fig, ax = plt.subplots()

n, bins, patches = ax.hist(list_hist_x, bin, density=False)


"""""
n_n = list(n)
bins_n = list(bins)
dictionary = {}
for i in range(bin):
    dictionary[bins_n[i]] = n_n[i]/N

s = pd.Series(dictionary)
s.plot.hist(alpha=0.6)
plt.show()
"""

y = []
bins_x = []
for i in bins:
    #if i < -0.1 or i > 0.1:
        bins_x.append(i)
        y.append(abs(1/(a*i))/(len(list_hist_x)/N))
ax.plot(bins_x, y, '--')
ax.set_xlabel('x')
ax.set_ylabel('num')
ax.set_title(r'Histogram')


# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()



list_x = []
list_y = []
for i in set(list_X):
    list_x.append(i)
    list_y.append(abs(1/(a*i)))

list_hist_x = []
for y in list_Y:
    if (y >= -0.2) and (y <= 0.2):
        list_hist_x.append(list_X[list_Y.index(y)])

def graph_PDF_2(list_cell, color, label):
    s = pd.Series(list_cell)
    s.plot.kde(color=color, label=label)

fg, ax = plt.subplots()
dx = (max(list_hist_x) - min(list_hist_x))/bin
plt.hist(list_hist_x, bins=20, density=True)
a = plt.hist(list_hist_x, bins=20)[0] * (1/N)

plt.show()

#fg, ax = plt.subplots()
graph_PDF_2(list_y, 'r', '')
#plt.scatter(list_x, list_y, marker="o", color="r")
plt.show()

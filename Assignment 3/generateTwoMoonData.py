'''
Generates two moon dataset

Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

def genTwoMoons(n=400, radius=1.5, width=0.5, dist=-1):
    rho = radius-width/2 + width*np.random.rand(1,n)
    phi = np.pi*np.random.randn(1,n)
    X = np.zeros((2,n))
    X[0], X[1] = polar2cart(rho, phi)
    id = X[1]<0
    X[0,id] = X[0,id] + radius
    X[1,id] = X[1,id] - dist
    Y = np.zeros(n)
    Y[id] = 1
    return X, Y


def polar2cart(rho, phi):
    x =  rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def main():
    X, Y = genTwoMoons(n=400, radius=1.5, width=0.5, dist=-1)
    colors = ['red','blue']
    fig = plt.figure(figsize=(6,4))
    plt.scatter(X[0], X[1], c=Y, cmap=clr.ListedColormap(colors))
    plt.show()
    
if __name__ == "__main__":
    main()
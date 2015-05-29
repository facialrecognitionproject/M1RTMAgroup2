# -*- coding: utf-8 -*-
"""
Created on Tue May 19 14:55:05 2015

@author: atoilha
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
	

#passage de l'image couleur en niveaux de gris
def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
	
img = mpimg.imread('Never_Giv_Up.jpeg')     
gray = rgb2gray(img)    
plt.imshow(gray, cmap = plt.get_cmap('gray'))
#plt.savefig('Never_Giv_Upgrey.jpeg')
#plt.show()

X = np.array(gray)
print X

#calcul de la moyenne de l'image

def computeMeanImage(X):
    #fonction qui calcule l'image des images d'apprentissage
    meanImage= np.mean(X)
    return meanImage
MU= computeMeanImage(X)
print MU


#centrage de l'image

def computeCentredImage(X, MU):
    #fonction qui centre les images de la base
    [irow,icol] = X.shape
    for i in range(len(irow)):
        for j in range(len(icol)):
            X=X-MU
    return X
X=X-MU
print 'la matrice d image centré :\n',X


#calcul de la matrice de correlation
def correlationMatrix(X):
    matrice_corr = np.cov(X)
    print 'matrice de correlation :\n', matrice_corr
    matrice_corre_norm=np.corrcoef( X,rowvar=0)
    matrice_corre_norm.shape
    print 'matrice de correlation normalise:\n', matrice_corre_norm
    return matrice_corr

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 10:25:07 2015

@author: aramaa
"""

import sys
sys.path.append("/home/SP2MI/gsadel01/Documents/eigenfaces")
# import numpy and matplotlib colormaps
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import os, sys, errno
import PIL.Image as Image



def read_img(path, sz=None):
	c = 0
	X,y = [], []
	for dirname, dirnames, filenames in os.walk(path):
		for subdirname in dirnames:
			subject_path = os.path.join(dirname, subdirname)
			for filename in os.listdir(subject_path):
				try:
					im = Image.open(os.path.join(subject_path, filename))
					im = im.convert("L")
					# resize to given size (if given)
					if (sz is not None):
						im = im.resize(sz, Image.ANTIALIAS)
					X.append(np.asarray(im, dtype=np.uint8))
					y.append(c)
				except IOError:
					print "I/O error({0}): {1}".format(errno, os.strerror)
				except:
					print "Unexpected error:", sys.exc_info()[0]
					raise
			c = c+1
	return [X,y]
 
 
[X,y] = read_img("/home/SP2MI/gsadel01/Documents/eigenfaces/IMG_BDD")
A=X[1];
#print(A)
#print(X)
#print(y)

def matrixToVector(A):
    B=np.hstack(A);
    return B
    
def vectorsToIMatrix(X,):
    n=X.length;
    imageMatrix=[];
    for i in range (n):
        imageMatrix.append=[matrixToVector(X[i])];
    return imageMatrix
    
    
#fonction qui calcule l'image moyenne des images d'apprentissage
#Cette fonction calcule l'image moyenne en fonction de toutes les images de la base de données
#le calcul de la moyenne permettra de faire une comparaison au niveau de  la phase d'identification
#cette fonction retourne MU : variable correspondant au resultat de la moyenne des images
def computeMeanImage(X):
    meanImage= np.mean(X)
    return meanImage
#MU= computeMeanImage(X)
#print MU

 #fonction qui centre les images de la base
#cette fonction permet de realiser le centrage de chaque image
#c'est-à-dire que pour chaque image ,l'image moyenne sera soustraite
##cette fonction retourne X : variable correspondant au resultat de l'image centrée
def computeCentredImage(X, MU):
##    [irow,icol] = X.shape
#    for i in range(irow):
#        for j in range(icol):
    X=X-MU
    return X
#X=X-MU
#print X
    
#fonction permettant d'afficher les valeurs propres et les vecteurs propres   
def printEigenComponent(EigeinValues, EigenVectors):
    for idx, val in enumerate(EigeinValues):
        print 'Lambda{0} = {1:.3f}'.format(idx, val)
        print 'b{0} : {1}.T\n'.format(idx, EigenVectors[:,idx])


#foction qui calcule les éléments propres de la matrice de correlation
#elle calcule les eigenvalues qui sont les valeurs propres et les eigenvectors qui sont les vecteurs propres
def computeEigenComponent(matrice_corre_norm):
    EigeinValues, EigenVectors = np.linalg.eig(matrice_corre_norm)
    printEigenComponent(EigeinValues, EigenVectors)
    return EigeinValues, EigenVectors
    
# Cette fonction nous permettra de calculer sla matrice de correlation
#A partir de la matrice de correlation,les valeurs propres et vecteurs 
#propres pourront etre calculés    
def matriceCorr(X):
    matrice_corr = np.cov(X)
    print 'matrice de correlation :\n', matrice_corr
    return matriceCorr    
#a= matriceCorr(X)
   
#cette fonction calcule la matrice de correlation normalisée
def matriceCorrNorm(X):
    matrice_corr_norm=np.corrcoef( X,rowvar=0)
    matrice_corr_norm.shape
    print 'matrice de correlation normalise:\n', matrice_corr_norm
#    print 'dimensions de la matrice de correlation normalise:\n', matrice_correlation_norm.shape
    return matrice_corr_norm
    
#c=matriceCorrNorm(X)
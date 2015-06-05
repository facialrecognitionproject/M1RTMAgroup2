# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 10:25:07 2015

@author: aramaa
"""

import sys
sys.path.append("C:\Users\Guy Florent\Documents\GitHub\M1RTMAgroup2\Programmation")
# import numpy and matplotlib colormaps
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import os, sys, errno
import PIL.Image as Image


#fonction permettant de récupérer toutes les images en parcourant la base de données répartie en dossier et sous dossier par sujet
#elle télécharge les images en liste
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
 
 



#fonction permettant de transformer les listes ou matrices d'images en vecteur répartis en colonnes dans une matrice
def asRowMatrix(X):
	if len(X) == 0:
		return np.array([])
	mat = np.empty((0, X[1].size), dtype=X[1].dtype)
	for row in X:
		mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
          
	return mat





#fonction qui calcule l'image moyenne des images d'apprentissage
#Cette fonction calcule l'image moyenne en fonction de toutes les images de la base de données
#le calcul de la moyenne permettra de faire une comparaison au niveau de  la phase d'identification
#cette fonction retourne MU : variable correspondant au resultat de la moyenne des images

def computeMeanImage(matImg):
    meanImage=np.mean(matImg, axis=0)
    return meanImage


##
# #fonction qui centre les images de la base
##cette fonction permet de realiser le centrage de chaque image
##c'est-à-dire que pour chaque image ,l'image moyenne sera soustraite
###cette fonction retourne X : variable correspondant au resultat de l'image centrée

def computeCentredImage(q, p):
    matImg=q-p
    return matImg


    
##fonction permettant d'afficher les valeurs propres et les vecteurs propres   
def printEigenComponent(EigeinValues, EigenVectors):
    for idx, val in enumerate(EigeinValues):
        print 'Lambda{0} = {1:.3f}'.format(idx, val)
        print 'b{0} : {1}.T\n'.format(idx, EigenVectors[:,idx])

#
##foction qui calcule les éléments propres de la matrice de correlation
##elle calcule les eigenvalues qui sont les valeurs propres et les eigenvectors qui sont les vecteurs propres
def computeEigenComponent(S):
    EigeinValues, EigenVectors = np.linalg.eig(S)
    printEigenComponent(EigeinValues, EigenVectors)
    return EigeinValues, EigenVectors

    
## Cette fonction nous permettra de calculer sla matrice de correlation
##A partir de la matrice de correlation,les valeurs propres et vecteurs 
##propres pourront etre calculés    
def matrixCorr(A):
    matrice_corr = np.cov(A)
    return matrice_corr    

#   
##cette fonction calcule la matrice de correlation normalisée
def matrixCov(H):
#    matrice_corr_norm = np.dot(H,H.T)
    matrice_cov=np.dot(H,H.T)
#    print 'dimensions de la matrice de correlation normalise:\n', matrice_correlation_norm.shape
    return matrice_cov
    
##correlMatrixNorm = matrixCorrNorm(CenImg)
##print 'matrice de correlation normalise:\n', correlMatrixNorm
#correlMatrixNorm = np.dot( CenImg,CenImg.T)
#print 'matrice de correlation normalise:\n', correlMatrixNorm
##EigeinValues, EigenVectors = computeEigenComponent(correlMatrixNorm)
def analyseComponents():
    print '****************** PrincipalesanalyseComponents ******************'


    Subjects = np.array(['S '+str(num) for num in np.arange(1,399)])

    matImg=asRowMatrix(X)
 
    print 'matrices image :\n', matImg
    
    MU= computeMeanImage(matImg)
    CenImg=computeCentredImage(matImg, MU)
    print 'Moyenne de limage  :\n',MU
    print 'Image centré  :\n',CenImg

    
    
    
#    correlMatrix= matrixCorr(CenImg)
    covMatrix = matrixCov(CenImg)
    covMatrix.shape
#    print 'matrice de correlation :\n', correlMatrix
    print 'matrice de covariancee:\n', covMatrix
    print 'dimensions de la matrice de covariancee:\n', covMatrix.shape
    EigeinValues, EigenVectors=computeEigenComponent(covMatrix)
    
    
    
    
#    nvx_echantillons=np.array(['jus de pomme','coca cola','C11'])
#    nvx_matImg=asRowMatrix(X)
#          
#    nvx_CenImg=computeCentredImage(nvx_matImg, MU)
#    print 'nvx_notes_c :\n', nvx_CenImg
       
    
    axe1 = EigenVectors[:,0]
    axe2 = EigenVectors[:,1]
#    axe3 = EigeinValues[:,3]

    plt.figure()
#    plt.subplot(1,2,1)
#    circle = plt.Circle((0,0), radius=1.0, fill=None, linewidth=1.5, edgecolor='black')
#    plt.gca().add_patch(circle)
#    for idx, description in enumerate(descriptions):
#        norme = np.sqrt(np.sum(axe1[idx]**2 + axe2[idx]**2))
#        plt.quiver(0, 0, axe1[idx]/norme, axe2[idx]/norme, color='r', units='xy', angles='xy', width=0.01, scale=1)
#        plt.quiver(0, 0, axe1[idx], axe2[idx], color='b', units='xy', angles='xy', width=0.01, scale=1)
#        plt.gca().text(0.6*axe1[idx]/norme, 0.6*axe2[idx]/norme, str(description), color='black', ha='left', va='bottom', fontweight='bold', size=13)
#    plt.grid(True)
#    plt.xlabel('Axe 1')
#    plt.ylabel('Axe 2')
#    plt.axis([-1, 1, -1, 1])
#
#    plt.subplot(1,2,2)
    projections_x = np.dot(covMatrix, axe1)
    projections_y = np.dot(covMatrix, axe2)
#    projections_z = np.dot(CenImg, axe3)
#    nvx_projections_x = np.dot(nvx_notes_c, axe1)
#    nvx_projections_y = np.dot(nvx_notes_c, axe2)
    plt.plot(projections_x, projections_y, 'ro', linewidth=0.5)
#    plt.plot(nvx_projections_x, nvx_projections_y, 'ro', linewidth=0.5)
    for idx, subj in enumerate(Subjects):
        plt.gca().text(projections_x[idx], projections_y[idx], subj, color='black', ha='left', va='bottom', fontweight='bold')
    plt.grid(True)
    plt.xlabel('Axe 1')
    plt.ylabel('Axe 2')
#    plt.axis([4.0, 6.0, 2.0, 3.6])
#    plt.axis([-3.5, 3.5, -3.5, 3.5])
    plt.show()
    
[X,y] = read_img("C:\Users\Guy Florent\Documents\GitHub\M1RTMAgroup2\Programmation\dataBaseImages")    

matImg=asRowMatrix(X)
print matImg

MU= computeMeanImage(matImg)
print 'Moyenne de limage  :\n',MU

CenImg=computeCentredImage(matImg, MU)
print 'Image centré  :\n',CenImg

correlMatrix= matrixCorr(CenImg)
print 'matrice de correlation :\n', correlMatrix  

  
def main(argv):
    plt.close('all')
    np.set_printoptions(precision=3, suppress=True)
#
    analyseComponents()
#    
#
if __name__ == '__main__':
    main(sys.argv)

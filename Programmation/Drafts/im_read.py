# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:14:07 2015

@author: gsadel01
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
#A=X[1];
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
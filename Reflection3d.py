# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

def column(matrix, i):
    return [row[i] for row in matrix]

def refFace(x0, y0, z0, x_box, y_box, z_box, side):
    if(side == 0):
        newPt = [x_box + (x_box - x0), y0, z0]
    if(side == 1):
        newPt = [-x_box + (-x_box - x0), y0, z0]
    if(side == 2):
        newPt = [x0, y_box + (y_box - y0), z0]
    if(side == 3):
        newPt = [x0, -y_box + (-y_box - y0), z0]
    if(side == 4):
        newPt = [x0, y0, z_box + (z_box - z0)]
    if(side == 5):
        newPt = [x0, y0, -z_box + (-z_box - z0)]
    return newPt

def reflectionStart(x0, y0, z0, x_box, y_box, z_box, types):
    refPts = []
    faces = [0, 1, 2, 3, 4, 5]
    order = 1
    refPts.append(refFace(x0, y0, z0, x_box, y_box, z_box, 0))
    refPts.append(refFace(x0, y0, z0, x_box, y_box, z_box, 1))
    refPts.append(refFace(x0, y0, z0, x_box, y_box, z_box, 2))
    refPts.append(refFace(x0, y0, z0, x_box, y_box, z_box, 3))
    refPts.append(refFace(x0, y0, z0, x_box, y_box, z_box, 4))
    refPts.append(refFace(x0, y0, z0, x_box, y_box, z_box, 5))
    return refPts, faces, order, types

def newRefs(Pts, x_box, y_box, z_box, faces, order, types, charges):
    xs = column(Pts, 0)
    ys = column(Pts, 1)
    zs = column(Pts, 2)
    numPts = len(xs)
    refPts = []
    facesNew = []
    chargesNew = []
    order += 1
    for i in range(0, numPts):
        for j in range(0, 6):
            if(j != faces[i]):
                refPts.append(refFace(xs[i], ys[i], zs[i], x_box, y_box, z_box, j))
                facesNew.append(j)
                if(types[j] == True):
                    charge = charges[i]
                else:
                    charge = not charges[i]
                chargesNew.append(charge)
    return refPts, facesNew, order, chargesNew


def makeRefs(x_0, y_0, z_0, x_box, y_box, z_box, maxOrder, types):
    RefPts, faces, order, charges = reflectionStart(x_0, y_0, z_0, x_box, y_box, z_box, types)
    AllRefs = RefPts
    Allfaces = faces
    Allords = [order]*len(faces)
    AllCharges = charges
    while(order < maxOrder):
        RefPts, faces, order, charges = newRefs(RefPts, x_box, y_box, z_box, faces, order, types, charges)
        for i in range(len(RefPts)):
            AllRefs.append(RefPts[i])
            Allfaces.append(faces[i])
            AllCharges.append(charges[i])
            Allords.append(order) 
        AllRefs, Allfaces, Allords, AllCharges = deleteRedundant(AllRefs, Allfaces, Allords, AllCharges)
    return AllRefs, Allfaces, Allords, AllCharges

def deleteRedundant(AllRefs, Allfaces, Allords, AllCharges):
    look = True
    i = 0
    while(look):
        Keep = [True]*len(AllRefs)
        for j in range(len(AllRefs)):
            if(AllRefs[j] == AllRefs[i]):
                samebool = (j == i)
                if(not samebool):
                    Keep[j] = False
        AllRefs = [b for a, b in zip(Keep, AllRefs) if a]
        Allfaces = [b for a, b in zip(Keep, Allfaces) if a]
        Allords = [b for a, b in zip(Keep, Allords) if a]
        AllCharges = [b for a, b in zip(Keep, AllCharges) if a]
        i += 1
        if(i >= len(AllRefs)):
            look = False
    return AllRefs, Allfaces, Allords, AllCharges

#Get rid of base point if it appeared as a reflection
def deleteBase(AllRefs, Allfaces, Allords, AllCharges, x0, y0, z0):
    Keep = [True]*len(AllRefs)
    for j in range(len(AllRefs)):
        if(AllRefs[j] == [x0, y0, z0]):
            Keep[j] = False
    AllRefs = [b for a, b in zip(Keep, AllRefs) if a]
    Allfaces = [b for a, b in zip(Keep, Allfaces) if a]
    Allords = [b for a, b in zip(Keep, Allords) if a]
    AllCharges = [b for a, b in zip(Keep, AllCharges) if a]
    return AllRefs, Allfaces, Allords, AllCharges

#For commented version in 2D, see ReflectionMaker.py
x_0 = 0
y_0 = 0
z_0 = 0
x_box = 1
y_box = 1
z_box = 1
maxOrder = 5
types = [True, True, True, True, True, False]
     
z_offset = z_box/2
z_0 = z_0 - z_offset     
AllRefs, Allfaces, Allords, AllCharges =  makeRefs(x_0, y_0, z_0, x_box, y_box, z_box, maxOrder, types)
AllRefs, Allfaces, Allords, AllCharges = deleteRedundant(AllRefs, Allfaces, Allords, AllCharges)
AllRefs, Allfaces, Allords, AllCharges = deleteBase(AllRefs, Allfaces, Allords, AllCharges, x_0, y_0, z_0)
for i in range(0, len(AllRefs)):
    AllRefs[i][2] += z_offset
z_0 = z_0 + z_offset

fig, ax = plt.subplots(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_0, y_0, z_0)
AllXs = column(AllRefs, 0)
AllYs = column(AllRefs, 1)
AllZs = column(AllRefs, 2)
ax.scatter(AllXs, AllYs, AllZs)
plt.show()

print('Coordinates of Reflected Point')
print(AllRefs)
print('Order of reflection')
print(Allords)
print('Reflection Charge')
print(AllCharges)

fig, ax = plt.subplots(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_0, y_0, z_0)
posXs = [b for a, b in zip(AllCharges, AllXs) if a]
posYs = [b for a, b in zip(AllCharges, AllYs) if a]
posZs = [b for a, b in zip(AllCharges, AllZs) if a]
negXs = [b for a, b in zip(AllCharges, AllXs) if not a]
negYs = [b for a, b in zip(AllCharges, AllYs) if not a]
negZs = [b for a, b in zip(AllCharges, AllZs) if not a]
ax.scatter(posXs, posYs, posZs)
ax.scatter(negXs, negYs, negZs)
plt.show()
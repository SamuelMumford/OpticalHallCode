# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:37:39 2021

@author: sammy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx

nR = 250
nOme = 250
z = 0
Dxx = 1
Dzz = 1
minO = 100
maxO = 50000
minR = .05
maxR = 3

def makeReIm(nR, nOme, z, Dxx, Dzz, minO, maxO, minR, maxR):
    omes = np.linspace(minO, maxO, nOme).reshape(1, nOme)
    ds = np.linspace(minR, maxR, nR).reshape(nR, 1)
    dEq = np.sqrt(ds**2/Dxx + z**2/Dzz)
    AmpE = np.exp(-np.matmul(dEq, np.sqrt(omes/2)))/dEq
    reP = np.cos(np.matmul(dEq, np.sqrt(omes/2)))
    imP = -np.sin(np.matmul(dEq, np.sqrt(omes/2)))
    real = AmpE*reP
    imag = AmpE*imP
    return real, imag

def phaseB(nR, nOme, Dxx, minO, maxO, minR, maxR):
    omes = np.linspace(minO, maxO, nOme).reshape(1, nOme)
    ds = np.linspace(minR, maxR, nR).reshape(nR, 1)
    dEq = np.sqrt(ds**2/Dxx)
    phaseBase = -np.matmul(dEq, np.sqrt(omes/2))
    return phaseBase

r, im = makeReIm(nR, nOme, z, Dxx, Dzz, minO, maxO, minR, maxR)
phase = np.arctan(im/r)

pB = phaseB(nR, nOme, Dxx, minO, maxO, minR, maxR)

pDiff = np.arctan(np.sin(phase - pB)/np.cos(phase - pB))
print(np.amax(np.abs(pDiff/pB)))

xs = np.linspace(minR, maxR, nR)
omegas = np.linspace(minO, maxO, nOme)
fig = plt.gcf()
plt.contourf(xs, 2*np.pi*np.sqrt(np.divide(2,omegas)), (pDiff/pB).T, 100, cmap = cmx.jet)
plt.colorbar()
plt.title('Relative Phase Error: Infinite Sample')
plt.xlabel('x-coordinate (u)')
plt.ylabel('Lambda (u)')
plt.show()

nH = 10
hs = np.linspace(.1, 1, nH)
layers = 7
for j in range(0, nH):
    h = hs[j]
    ind = -1
    imT = np.copy(im)
    reT = np.copy(r)
    for i in range(0, layers):
        z = 2*(i + 1)*h
        rT, iT = makeReIm(nR, nOme, z, Dxx, Dzz, minO, maxO, minR, maxR)
        reT += ind*rT
        imT += ind*iT
        ind = ind*-1
    phase = np.arctan(imT/reT)
    pDiff = np.arctan(np.sin(phase - pB)/np.cos(phase - pB))
    print(np.amax(np.abs(pDiff/pB)))
    # fig = plt.gcf()
    # plt.contourf(xs, 2*np.pi*np.sqrt(np.divide(2,omegas)), pDiff.T, 100, cmap = cmx.jet)
    # plt.colorbar()
    # plt.title('Phase Error: h=' + str(np.round(100*h)/100))
    # plt.xlabel('x-coordinate (u)')
    # plt.ylabel('Lambda (u)')
    # plt.show()
    
    fig = plt.gcf()
    plt.contourf(xs, 2*np.pi*np.sqrt(np.divide(2,omegas)), (pDiff/pB).T, 200, cmap = cmx.jet)
    plt.colorbar()
    plt.title('Relative Phase Error: h=' + str(np.round(100*h)/100))
    plt.xlabel('x-coordinate (u)')
    plt.ylabel('Lambda (u)')
    plt.show()
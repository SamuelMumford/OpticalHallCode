# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:37:39 2021

@author: sammy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx

#The number of spacial and frequency points to examine
nR = 250
nOme = 250
#Range of space and frequencies
minO = 100
maxO = 50000
minR = .05
maxR = 3
#Initial sample parameters
z = 0
Dxx = 1
Dzz = 1

#Find the real and imaginary parts of the response to a source centered
#at (0, 0, z). Done for a range of frequencies and in-plane points.
def makeReIm(nR, nOme, z, Dxx, Dzz, minO, maxO, minR, maxR):
    #List of frequencies and in-plane distances as a row and coloumn vector
    omes = np.linspace(minO, maxO, nOme).reshape(1, nOme)
    ds = np.linspace(minR, maxR, nR).reshape(nR, 1)
    #Convert to a diffusivity-scaled distance from the source
    dEq = np.sqrt(ds**2/Dxx + z**2/Dzz)
    #Make a matrix of amplitudes of the source response
    #Element (i, j) corresponds to (distance i, frequency j)
    AmpE = np.exp(-np.matmul(dEq, np.sqrt(omes/2)))/dEq
    #Find how much of the amplitude is in the real and imaginary parts
    reP = np.cos(np.matmul(dEq, np.sqrt(omes/2)))
    imP = -np.sin(np.matmul(dEq, np.sqrt(omes/2)))
    #Split response into real and imaginary parts
    real = AmpE*reP
    imag = AmpE*imP
    return real, imag

#Find the base phase response for an ideal source centered at (0, 0, 0)
#Used to find how much phase changes with mirror charges/sample height
def phaseB(nR, nOme, Dxx, minO, maxO, minR, maxR):
    omes = np.linspace(minO, maxO, nOme).reshape(1, nOme)
    ds = np.linspace(minR, maxR, nR).reshape(nR, 1)
    dEq = np.sqrt(ds**2/Dxx)
    phaseBase = -np.matmul(dEq, np.sqrt(omes/2))
    return phaseBase

#Find the real and imaginary parts of the response to an beam centered at (0,0,0)
#for a set of spacial points and frequencies
r, im = makeReIm(nR, nOme, z, Dxx, Dzz, minO, maxO, minR, maxR)
#Convert to phase
phase = np.arctan(im/r)
#Find the real and imaginary parts of the response to a beam centered at (0,0,0)
#using the naive sqrt(omega)*r model
pB = phaseB(nR, nOme, Dxx, minO, maxO, minR, maxR)
#Find the phase difference between the model with 0 mirror charges and 
#the naive model. It should be 0. The arctan takes care of pi phase differences
#in angle definitions (which can make rounding a problem)
pDiff = np.arctan(np.sin(phase - pB)/np.cos(phase - pB))
#Print the largest difference between the calculaed phase difference and the naive
#model
print(np.amax(np.abs(pDiff/pB)))

#Plot it, should just be rounding errors
xs = np.linspace(minR, maxR, nR)
omegas = np.linspace(minO, maxO, nOme)
fig = plt.gcf()
plt.contourf(xs, 2*np.pi*np.sqrt(np.divide(2,omegas)), (pDiff/pB).T, 100, cmap = cmx.jet)
plt.colorbar()
plt.title('Relative Phase Error: Infinite Sample')
plt.xlabel('x-coordinate (u)')
plt.ylabel('Lambda (u)')
plt.show()

#Define the range of heights for mirror charges and the order of mirror
#reflection to go up to
nH = 10
hs = np.linspace(.1, 1, nH)
layers = 7
#Calculate the phase difference for each height
for j in range(0, nH):
    h = hs[j]
    #The first mirror charge has opposite sign
    ind = -1
    #Start with the base response from a source centered at (0,0,0)
    imT = np.copy(im)
    reT = np.copy(r)
    #For each mirror charge layer, add the contribution from the reflected
    #charges to the total real and imaginary part
    for i in range(0, layers):
        #Find the response to a source at i+1 mirror order
        z = 2*(i + 1)*h
        rT, iT = makeReIm(nR, nOme, z, Dxx, Dzz, minO, maxO, minR, maxR)
        #Add or subtract that response based on parity of mirror order
        reT += ind*rT
        imT += ind*iT
        ind = ind*-1
    #Find the phase including mirror charges
    phase = np.arctan(imT/reT)
    #Compare phase to the naive case
    pDiff = np.arctan(np.sin(phase - pB)/np.cos(phase - pB))
    print(np.amax(np.abs(pDiff/pB)))
    
    #Plot the phase difference
    fig = plt.gcf()
    plt.contourf(xs, 2*np.pi*np.sqrt(np.divide(2,omegas)), (pDiff/pB).T, 200, cmap = cmx.jet)
    plt.colorbar()
    plt.title('Relative Phase Error: h=' + str(np.round(100*h)/100))
    plt.xlabel('x-coordinate (u)')
    plt.ylabel('Lambda (u)')
    plt.show()
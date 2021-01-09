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

#Find the real and imaginary parts of the response to an beam centered at (0,0,0)
#for a set of spacial points and frequencies
r, im = makeReIm(nR, nOme, z, Dxx, Dzz, minO, maxO, minR, maxR)
#Convert to amplitude
Amp = np.sqrt(r**2 + im**2)

#Plot it
xs = np.linspace(minR, maxR, nR)
omegas = np.linspace(minO, maxO, nOme)
fig = plt.gcf()
plt.contourf(xs, 2*np.pi*np.sqrt(np.divide(2,omegas)), (Amp).T, 100, cmap = cmx.jet)
plt.colorbar()
plt.title('Response Amplitude: Infinite Sample')
plt.xlabel('x-coordinate (u)')
plt.ylabel('Lambda (u)')
plt.show()

#Define the range of heights for mirror charges and the order of mirror
#reflection to go up to
nH = 10
hs = np.linspace(.1, 1, nH)
layers = 15
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
    AmpP = np.sqrt(reT**2 + imT**2)
    
    #Plot the amplitude difference
    fig = plt.gcf()
    plt.contourf(xs, 2*np.pi*np.sqrt(np.divide(2,omegas)), ((AmpP - Amp)/(Amp)).T, 200, cmap = cmx.jet)
    plt.colorbar()
    plt.title('Fractional Change in Amplitude: h=' + str(np.round(100*h)/100))
    plt.xlabel('x-coordinate (u)')
    plt.ylabel('Lambda (u)')
    plt.show()
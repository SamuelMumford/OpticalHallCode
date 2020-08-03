
#work in progress 
from dolfin import *
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import matplotlib.cm as cmx
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
from scipy.optimize import curve_fit
import pylab as plt
import cmath

radius = .2
spacing = .01
#penetration of laser heat source
b = 100
depth = 1/b
alpha1 = 1
ktrue = 1000
omega = 200
n=1
tol = 1E-14
sp = .005
Dx = 1
Dy = 1
Dz = 1
#Set Dxy1 near 0 for reference
Dxy1 = 0.05
Dyx1 = - Dxy1
#angle of phase cut (not in this version)
theta = np.pi/2
#truexc = 0
#trueyc = 0
#xcenter = truexc/np.sqrt(Dx)
#ycenter = trueyc/np.sqrt(Dy)
#dimensions of sample
truex = 1
truey = 1
truez = 1
xmax = truex/np.sqrt(Dx)
ymax = truey/np.sqrt(Dy)
zmax = truez/np.sqrt(Dz)
#number of lattice points in each direction
xpoints = 45
ypoints = 45
zpoints = 15
e = .1
beta = .89
#Amplitude of ideal solution
A=1

#Choose the center point of the beam, preferably not right on a lattice point to keep from dividing by zero
heatx = .3
heaty = 0
#heatz = same for all mirror points
heatz = zmax/2
#heat_sign: hot = 1, cold = -1 heating source
heat_sign = 1
#How many times to reflect. maxOrder of 1 will only make 4 mirror charges/one reflection
maxOrder = 4
#If a del T = 0 boundary (same sign), True. If a T=0 boundary (flip sign), false
#In the order +x, -x, +y, -y boundaries
types = [True, True, True, True]

def solver(g, xmax, ymax, zmax, tol, mesh, omega, alpha, Dxy, Dxx, Dyy):
    (ureal, uimag) = TrialFunctions(W)
    (v1, v2) = TestFunctions(W)
    if(Dxy == 0):
        a = (alpha*dot(grad(uimag),grad(v1)) + omega*ureal*v1 + omega*uimag*v2 - alpha*dot(grad(ureal),grad(v2)))*dx
    else:
        a = (alpha*dot(grad(uimag),grad(v1)) + omega*ureal*v1 + omega*uimag*v2 - alpha*dot(grad(ureal),grad(v2)))*dx + (Dxy/np.sqrt(Dxx*Dyy)*alpha*uimag.dx(1)*v1.dx(0) - Dxy/np.sqrt(Dxx*Dyy)*alpha*uimag.dx(0)*v1.dx(1) - Dxy/np.sqrt(Dxx*Dyy)*alpha*ureal.dx(1)*v2.dx(0) + Dxy/np.sqrt(Dxx*Dyy)*alpha*ureal.dx(0)*v2.dx(1))*dx
    L =  - g*v2*dx
    def bot(x, on_boundary): return on_boundary and near(x[2], -zmax/2, tol)
    noslip = Constant(0.0)
    bc0 = DirichletBC(W.sub(0), noslip, bot)
    bc1 = DirichletBC(W.sub(1), noslip, bot)
    G = Expression('0', degree = 0)
    def boundary(x, on_boundary):
            return on_boundary and not near(x[2], -zmax/2, tol)    
    bcs = [bc0, bc1]
    w = Function(W)
    solve(a == L, w, bcs, solver_parameters={'linear_solver':'mumps'})
    (ureal, uimag) = w.split()
    return ureal, uimag

def phasef(ureal, uimag, xline, yline, depth, z):
    zpoints = len(z)
    wtot = 0
    upointr = 0
    upointi = 0
    for i in range(0,zpoints):
        uliner = ureal(xline,yline,z[i])
        ulinei = uimag(xline,yline,z[i])
        wlayer = np.exp(z[i]/depth)#/ np.sum()
        wtot = wtot + wlayer
        upointr = upointr + uliner*wlayer
        upointi = upointi + ulinei*wlayer
    upointr = upointr/wtot
    upointi = upointi/wtot
    phase = np.arctan2(upointi, upointr)
    return phase

def linepoints(theta, truexc, trueyc, truex, truey, truez, spacing):
    d = 0.0
    listpoints = np.array([[d, truexc, trueyc]])
    lookright = True 
    while lookright:
        d = d+spacing
        xtemp = truexc + np.cos(theta)*d
        ytemp = trueyc + np.sin(theta)*d
        if xtemp > truex/2 or ytemp > truey/2 or xtemp < -truex/2 or ytemp < -truey/2:
            lookright = False
        else:
            listpoints = np.append(listpoints,[[d, xtemp, ytemp]], axis = 0)
    lookleft = True
    d = 0
    while lookleft:
        d = d - spacing
        xtemp = truexc + np.cos(theta)*d
        ytemp = trueyc + np.sin(theta)*d
        if xtemp > truex/2 or ytemp > truey/2 or xtemp < -truex/2 or ytemp < -truey/2:
            lookleft = False
        else:
            listpoints = np.append([[d, xtemp, ytemp]],listpoints, axis = 0)
    return listpoints
    
def checkradius(radius, truxc, trueyc, truex, truey, x, y):
    if (radius + truexc) > truex/2 or (radius + trueyc) > truey/2 or (radius - truexc) < -truex/2 or (radius - trueyc) < -truey/2:
            radius = min(np.array([truex/2 - truexc, truey/2 - trueyc]))
            print('Radius changed to fall within boundary')
            print('Radius = ' + str(radius))
    rpoints = np.zeros(2)
    xr = np.array([0])
    yr = np.array([0])
    thetal = np.linspace(0, 2*np.pi, 1000)
    for i in range (0, len(thetal)):
        theta = thetal[i]
        xpoint = radius*np.cos(theta) + truexc
        ypoint = radius*np.sin(theta) + trueyc
        xr = np.append(xr, xpoint)
        yr = np.append(yr, ypoint)
    return radius, xr, yr, thetal

def phaseline(ureal, uimag, xline, yline, depth, z):
    phasearray = np.zeros(len(xline))
    for i in range(0, len(xline)):
        phasearray[i] = phasef(ureal, uimag, xline[i], yline[i], depth, z)
    return phasearray
    
def phasecirc(ureal, uimag, xr, yr, depth, z):
    phasearray = np.zeros(len(xr)-1)
    for i in range(1, len(xr)):
        phasearray[i-1] = phasef(ureal, uimag, xr[i], yr[i], depth, z)
    return phasearray

def utotal(cutnumx, cutnumy, x, y, z, zpoints, ureal, uimag, depth): 
    wtot = 0
    utotr = np.zeros(cutnumy*cutnumx)
    utoti = np.zeros(cutnumy*cutnumx)
    for i in range(0,zpoints):
        #print(z[i])
        points = [(x_, y_, z[i]) for x_ in x for y_ in y]
        u_sheetr = np.array([ureal(point) for point in points])
        u_sheeti = np.array([uimag(point) for point in points])
        wlayer = np.exp(z[i]/depth)#/ np.sum()
        wtot = wtot + wlayer
        utotr = utotr + u_sheetr * wlayer
        utoti = utoti + u_sheeti * wlayer
    utotr = utotr/wtot
    utoti = utoti/wtot
    utotr = np.reshape(utotr, (cutnumx, cutnumy))
    utoti = np.reshape(utoti, (cutnumx, cutnumy))
    return utotr, utoti

#Find temperature at one point on the lattice, used to find values 2D temp plots
def Temp_at_point(xc, yc, zc, omega, A, x, y, z):
    r = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
    T = A/r*np.exp(-np.sqrt(1j*omega)*r)
    Treal = T.real
    Timag = T.imag
    return Treal, Timag

#2D Temp
def Ttotal(cutnumx, cutnumy, x, y, z, cutnumz, xc, yx, zc, depth, sign): 
    #changed zpoints to cutnumz
    wtot = 0
    Ttotr = np.zeros([cutnumy,cutnumx])
    Ttoti = np.zeros([cutnumy,cutnumx])
    xs, ys = np.meshgrid(x, y, indexing='xy')
    for i in range(0,cutnumz):
        T_sheetr = np.zeros([cutnumy,cutnumx])
        T_sheeti = np.zeros([cutnumy,cutnumx])
        for l in range(0,len(xc)):
            Trnew, Tinew = Temp_at_point(xc[l], yc[l], zc[l], omega, A, xs, ys, z[i])
            T_sheetr += sign[l]*Trnew
            T_sheeti += sign[l]*Tinew
        wlayer = np.exp(z[i]/depth)#/ np.sum()
        wtot = wtot + wlayer
        Ttotr = Ttotr + T_sheetr * wlayer
        Ttoti = Ttoti + T_sheeti * wlayer
    Ttotr = Ttotr/wtot
    Ttoti = Ttoti/wtot
    return np.transpose(Ttotr), np.transpose(Ttoti)


def scatter3d(meshpts, u):
    meshpts = mesh.coordinates()
    uarray = np.array([u(Point(x,y,z)) for x,y,z in meshpts])
    #print('uarray')
    #print(uarray)
    xdata = meshpts[:,0]
    xdata = np.transpose(xdata)
    ydata = meshpts[:,1]
    ydata = np.transpose(ydata)
    zdata = meshpts[:,2]
    zdata = np.transpose(zdata)
    cm = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=min(uarray), vmax=max(uarray))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xdata, ydata, zdata, c=scalarMap.to_rgba(uarray))
    scalarMap.set_array(uarray)
    fig.colorbar(scalarMap)
    ax.set_xlabel('x (u)')
    ax.set_ylabel('y (u)')
    ax.set_zlabel('z (u)')
    #plt.show()
    
def scatter3dphase(meshpts, uarray):
    meshpts = mesh.coordinates()
    #uarray = np.array([u(Point(x,y,z)) for x,y,z in meshpts])
    #print('uarray')
    #print(uarray)
    xdata = meshpts[:,0]
    xdata = np.transpose(xdata)
    ydata = meshpts[:,1]
    ydata = np.transpose(ydata)
    zdata = meshpts[:,2]
    zdata = np.transpose(zdata)
    cm = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=min(uarray), vmax=max(uarray))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xdata, ydata, zdata, c=scalarMap.to_rgba(uarray))
    scalarMap.set_array(uarray)
    fig.colorbar(scalarMap)
    plt.show()
    
def scatterT(meshpts, T):
    meshpts = mesh.coordinates()
    xdata = meshpts[:,0]
    xdata = np.transpose(xdata)
    ydata = meshpts[:,1]
    ydata = np.transpose(ydata)
    zdata = meshpts[:,2]
    zdata = np.transpose(zdata)
    cm = plt.get_cmap('jet')
    cNorm = matplotlib.colors.Normalize(vmin=min(T), vmax=max(T))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xdata, ydata, zdata, c=scalarMap.to_rgba(T))
    scalarMap.set_array(T)
    fig.colorbar(scalarMap)
    plt.show()
  
#Find the temperature at every point on the mesh (used only for 3d plots)
def Temp_all_mesh(xc, yc, zc, omega, A, mesh, sign):
    #Check if need to scale meshpts by D again
    meshpts = mesh.coordinates()
    xs = meshpts[:,0]
    ys = meshpts[:,1]
    zs = meshpts[:,2]
    for j in range (0, len(xc)):
        r = np.sqrt((xs - xc[j])**2 + (ys - yc[j])**2 + (zs - zc[j])**2)
        T = A/r*np.exp(-np.sqrt(1j*omega)*r)
        Treal = T.real
        Timag = T.imag
        if(j == 0):
            Tr = sign[j]*Treal
            Ti = sign[j]*Timag
        else:
            Tr += sign[j]*Treal
            Ti += sign[j]*Timag
    return Tr, Ti

#Find a column from a list, used to get the x and y data seperately
def column(matrix, i):
    return [row[i] for row in matrix]

#Reflect over a face listed by side to get new coordinates. Same +x, -x, +y, -y indexing as types
def refFace(x0, y0, x_box, y_box, side):
    if(side == 0):
        newPt = [x_box + (x_box - x0), y0]
    if(side == 1):
        newPt = [-x_box + (-x_box - x0), y0]
    if(side == 2):
        newPt = [x0, y_box + (y_box - y0)]
    if(side == 3):
        newPt = [x0, -y_box + (-y_box - y0)]
    return newPt

#Initialize the list of reflected points with reflections from the base source
#over each boundary
def reflectionStart(x0, y0, x_box, y_box, types):
    refPts = []
    faces = [0, 1, 2, 3]
    order = 1
    refPts.append(refFace(x0, y0, x_box, y_box, 0))
    refPts.append(refFace(x0, y0, x_box, y_box, 1))
    refPts.append(refFace(x0, y0, x_box, y_box, 2))
    refPts.append(refFace(x0, y0, x_box, y_box, 3))
    charges = types
    return refPts, faces, order, charges

#Take a list of reflected charge points and which face they were reflected over most recently
#and reflect those points over all the other faces
def newRefs(Pts, x_box, y_box, faces, order, types, charges):
    xs = column(Pts, 0)
    ys = column(Pts, 1)
    numPts = len(xs)
    refPts = []
    facesNew = []
    chargesNew = []
    order += 1
    for i in range(0, numPts):
        for j in range(0, 4):
            if(j != faces[i]):
                refPts.append(refFace(xs[i], ys[i], x_box, y_box, j))
                facesNew.append(j)
                if(types[j] == True):
                    charge = charges[i]
                else:
                    charge = not charges[i]
                chargesNew.append(charge)
    return refPts, facesNew, order, chargesNew

#Call newRefs repeatedly to make a full list of all the reflected points
def makeRefs(x_0, y_0, x_box, y_box, maxOrder, types):
    RefPts, faces, order, charges = reflectionStart(x_0, y_0, x_box, y_box, types)
    AllRefs = RefPts
    Allfaces = faces
    Allords = [order]*len(faces)
    AllCharges = charges
    while(order < maxOrder):
        RefPts, faces, order, charges = newRefs(RefPts, x_box, y_box, faces, order, types, charges)
        for i in range(len(RefPts)):
            AllRefs.append(RefPts[i])
            Allfaces.append(faces[i])
            Allords.append(order)
            AllCharges.append(charges[i])
        AllRefs, Allfaces, Allords, AllCharges = deleteRedundant(AllRefs, Allfaces, Allords, AllCharges)
    return AllRefs, Allfaces, Allords, AllCharges     

#Get rid of redundant points
def deleteRedundant(AllRefs, Allfaces, Allords, AllCharges):
    look = True
    i = 0
    while(look):
        Keep = [True]*len(AllRefs)
        for j in range(len(AllRefs)):
            pj = AllRefs[j]
            xj = pj[0]
            yj = pj[1]
            pI = AllRefs[i]
            xI = pI[0]
            yI = pI[1]
            dist = np.sqrt((xI - xj)**2 + (yI - yj)**2)
            if dist < 0.0001:
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
def deleteBase(AllRefs, Allfaces, Allords, AllCharges, x0, y0):
    Keep = [True]*len(AllRefs)
    for j in range(len(AllRefs)):
        #if(AllRefs[j] == [x0, y0]):
        pj = AllRefs[j]
        xj = pj[0]
        yj = pj[1]
        dist = np.sqrt((xj - x0)**2 + (yj - y0)**2)
        if dist < 0.0001:
            Keep[j] = False
            #if dist > 0:
    AllRefs = [b for a, b in zip(Keep, AllRefs) if a]
    Allfaces = [b for a, b in zip(Keep, Allfaces) if a]
    Allords = [b for a, b in zip(Keep, Allords) if a]
    AllCharges = [b for a, b in zip(Keep, AllCharges) if a]
    return AllRefs, Allfaces, Allords, AllCharges

def basePhase(x, y, x_0, y_0, Dx, Dy, omega):
    r = np.sqrt((x - x_0)**2/Dx + (y - y_0)**2/Dy)
    ph = -np.sqrt(omega/2)*r
    return ph
print('Running...')

def PS(x, y, z, xs, ys, zs, Dxx, Dyy, Dzz, omega):
    r = np.sqrt((x - xs)**2/Dxx + (y - ys)**2/Dyy + (z - zs)**2/Dzz)
    f = 1/r*np.exp(-np.sqrt(1j*omega)*r)
    return f

def lineOfCharge(x, y, z, xStart, xStop, yStart, yStop, zStart, zStop, sp, Dxx, Dyy, Dzz, omega):
    n = np.sqrt((xStart - xStop)**2 + (yStart - yStop)**2 + (zStart - zStop)**2)/sp
    n = int(round(n))
    xs = np.linspace(xStart, xStop, n)
    ys = np.linspace(yStart, yStop, n)
    zs = np.linspace(zStart, zStop, n)
    for i in range(0, n):
        if (i == 0):
            res = PS(x, y, z, xs[i], ys[i], zs[i], Dxx, Dyy, Dzz, omega)
        else:
            res += PS(x, y, z, xs[i], ys[i], zs[i], Dxx, Dyy, Dzz, omega)
    return res

def DxyBoundaryTerms(x, y, z, x_0, y_0, z_0, x_max, y_max, z_max, Dxx, Dyy, Dzz, Dxy, omega, A, sp):
    boundRes = lineOfCharge(x, y, z, 2*x_max - x_0, 4*x_max + x_0, y_0 + sp/2, y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
    boundRes -= lineOfCharge(x, y, z, 2*x_max - x_0, 4*x_max + x_0, y_0 - sp/2, y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
   
    boundRes += lineOfCharge(x, y, z, -2*x_max - x_0, -4*x_max + x_0, y_0 - sp/2, y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
    boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0, -4*x_max + x_0, y_0 + sp/2, y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
   
    boundRes += lineOfCharge(x, y, z, x_0 - sp/2, x_0 - sp/2, 2*y_max - y_0, 4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
    boundRes -= lineOfCharge(x, y, z, x_0 + sp/2, x_0 + sp/2, 2*y_max - y_0, 4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
   
    boundRes += lineOfCharge(x, y, z, x_0 + sp/2, x_0 + sp/2, -2*y_max - y_0, -4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
    boundRes -= lineOfCharge(x, y, z, x_0 - sp/2, x_0 - sp/2, -2*y_max - y_0, -4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
    
    addMirror = False
    
    if(addMirror):
        #print('not doing higher order, it did not work')
#Try 3
        boundRes -= lineOfCharge(x, y, z, 2*x_max - x_0, 4*x_max + x_0, 2*y_max - y_0 + sp/2, 2*y_max - y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes += lineOfCharge(x, y, z, 2*x_max - x_0, 4*x_max + x_0, 2*y_max - y_0 - sp/2, 2*y_max - y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0, x_0, 2*y_max - y_0 + sp/2, 2*y_max - y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes += lineOfCharge(x, y, z, -2*x_max - x_0, x_0, 2*y_max - y_0 - sp/2, 2*y_max - y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx

        boundRes -= lineOfCharge(x, y, z, 2*x_max - x_0, 4*x_max + x_0, -2*y_max - y_0 + sp/2, -2*y_max - y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes += lineOfCharge(x, y, z, 2*x_max - x_0, 4*x_max + x_0, -2*y_max - y_0 - sp/2, -2*y_max - y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0, x_0, -2*y_max - y_0 + sp/2, -2*y_max - y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes += lineOfCharge(x, y, z, -2*x_max - x_0, x_0, -2*y_max - y_0 - sp/2, -2*y_max - y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
    
        boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0, -4*x_max + x_0, 2*y_max - y_0 - sp/2, 2*y_max - y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes += lineOfCharge(x, y, z, -2*x_max - x_0, -4*x_max + x_0, 2*y_max - y_0 + sp/2, 2*y_max - y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes -= lineOfCharge(x, y, z, x_0, 2*x_max - x_0, 2*y_max - y_0 - sp/2, 2*y_max - y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes += lineOfCharge(x, y, z, x_0, 2*x_max - x_0, 2*y_max - y_0 + sp/2, 2*y_max - y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx

        boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0, -4*x_max + x_0, -2*y_max - y_0 - sp/2, -2*y_max - y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes += lineOfCharge(x, y, z, -2*x_max - x_0, -4*x_max + x_0, -2*y_max - y_0 + sp/2, -2*y_max - y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes -= lineOfCharge(x, y, z, x_0, 2*x_max - x_0, -2*y_max - y_0 - sp/2, -2*y_max - y_0 - sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx
        boundRes += lineOfCharge(x, y, z, x_0, 2*x_max - x_0, -2*y_max - y_0 + sp/2, -2*y_max - y_0 + sp/2, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dxx

        
        boundRes -= lineOfCharge(x, y, z, 2*x_max - x_0 - sp/2, 2*x_max - x_0 - sp/2, 2*y_max - y_0, 4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes += lineOfCharge(x, y, z, 2*x_max - x_0 + sp/2, 2*x_max - x_0 + sp/2, 2*y_max - y_0, 4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes -= lineOfCharge(x, y, z, 2*x_max - x_0 - sp/2, 2*x_max - x_0 - sp/2, y_0, -2*y_max - y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes += lineOfCharge(x, y, z, 2*x_max - x_0 + sp/2, 2*x_max - x_0 + sp/2, y_0, -2*y_max - y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy

        boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0 - sp/2, -2*x_max - x_0 - sp/2, 2*y_max - y_0, 4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes += lineOfCharge(x, y, z, -2*x_max - x_0 + sp/2, -2*x_max - x_0 + sp/2, 2*y_max - y_0, 4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0 - sp/2, -2*x_max - x_0 - sp/2, y_0, -2*y_max - y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes += lineOfCharge(x, y, z, -2*x_max - x_0 + sp/2, -2*x_max - x_0 + sp/2, y_0, -2*y_max - y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        
        boundRes += lineOfCharge(x, y, z, 2*x_max - x_0 - sp/2, 2*x_max - x_0 - sp/2, -2*y_max - y_0, -4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes -= lineOfCharge(x, y, z, 2*x_max - x_0 + sp/2, 2*x_max - x_0 + sp/2, -2*y_max - y_0, -4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes += lineOfCharge(x, y, z, 2*x_max - x_0 - sp/2, 2*x_max - x_0 - sp/2, y_0, 2*y_max - y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes -= lineOfCharge(x, y, z, 2*x_max - x_0 + sp/2, 2*x_max - x_0 + sp/2, y_0, 2*y_max - y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy

        boundRes += lineOfCharge(x, y, z, -2*x_max - x_0 - sp/2, -2*x_max - x_0 - sp/2, -2*y_max - y_0, -4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0 + sp/2, -2*x_max - x_0 + sp/2, -2*y_max - y_0, -4*y_max + y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes += lineOfCharge(x, y, z, -2*x_max - x_0 - sp/2, -2*x_max - x_0 - sp/2, y_0, 2*y_max - y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        boundRes -= lineOfCharge(x, y, z, -2*x_max - x_0 + sp/2, -2*x_max - x_0 + sp/2, y_0, 2*y_max - y_0, z_0, z_0, sp, Dxx, Dyy, Dzz, omega)/Dyy
        
    boundRes *= 2*A*Dxy
    return boundRes.real, boundRes.imag

def DxyTotal(cutnumx, cutnumy, x, y, z, cutnumz, x_0, y_0, z_0, xmax, ymax, zmax, depth, omega, Dxx, Dyy, Dzz, Dxy, A, sp): 
    #changed zpoints to cutnumz
    wtot = 0
    sign = 1
    Ttotr = np.zeros([cutnumy,cutnumx])
    Ttoti = np.zeros([cutnumy,cutnumx])
    xs, ys = np.meshgrid(x, y, indexing='xy')
    for i in range(0,cutnumz):
        T_sheetr = np.zeros([cutnumy,cutnumx])
        T_sheeti = np.zeros([cutnumy,cutnumx])
        Trnew, Tinew = DxyBoundaryTerms(xs, ys, z[i], x_0, y_0, z_0, xmax, ymax, zmax, Dxx, Dyy, Dzz, Dxy, omega, A, sp)
        T_sheeti += sign*Tinew
        T_sheetr += sign*Trnew
        wlayer = np.exp(z[i]/depth)#/ np.sum()
        wtot += wlayer
        Ttotr += T_sheetr * wlayer
        Ttoti += T_sheeti * wlayer
    Ttotr = Ttotr/wtot
    Ttoti = Ttoti/wtot
    return np.transpose(Ttotr), np.transpose(Ttoti)

def delOffset(image_unwrapped1):
    imax1 = max(image_unwrapped1.max(axis = 1))
    n = np.abs(imax1)//(np.pi*2)*np.sign(imax1)
    image_unwrapped1 = image_unwrapped1 - 2*np.pi*n
    return image_unwrapped1

def lineTemp(lineXs, lineYs, z, cutnumz, x_0, y_0, z_0, xc, yc, zc, sign, s0, xmax, ymax, zmax, depth, omega, Dxx, Dyy, Dzz, Dxy, A, sp):
#changed zpoints to cutnumz
    wtot = 0
    Ttotr = np.zeros(len(lineXs))
    Ttoti = np.zeros(len(lineXs))
    for i in range(0,cutnumz):
        T_sheetr = np.zeros(len(lineXs))
        T_sheeti = np.zeros(len(lineXs))
        Dr, Di = DxyBoundaryTerms(lineXs, lineYs, z[i], x_0, y_0, z_0, xmax, ymax, zmax, Dxx, Dyy, Dzz, Dxy, omega, A, sp)
        T_sheeti += s0*Di
        T_sheetr += s0*Dr
        for l in range(0, len(xc)):
            Trnew, Tinew = Temp_at_point(xc[l], yc[l], zc[l], omega, A, lineXs, lineYs, z[i])
            T_sheeti += sign[l]*Tinew
            T_sheetr += sign[l]*Trnew
        wlayer = np.exp(z[i]/depth)
        wtot += wlayer
        Ttotr += T_sheetr * wlayer
        Ttoti += T_sheeti * wlayer
    Ttotr = Ttotr/wtot
    Ttoti = Ttoti/wtot
    return Ttotr, Ttoti

#Find mirror points

#Coordinates of the base heat source
x_0 = heatx
y_0 = heaty
#Length and Width of the box centered at (0,0)
x_box = xmax/2
y_box = ymax/2

#Find the location of all the reflections, then delete redundant points
#Also return which face was reflected over to get each point, how many reflections were
#done to get each point, and the sign of that mirror charge (True is same sign as original)
AllRefs, Allfaces, Allords, AllCharges = makeRefs(x_0, y_0, x_box, y_box, maxOrder, types)
AllRefs, Allfaces, Allords, AllCharges = deleteRedundant(AllRefs, Allfaces, Allords, AllCharges)
AllRefs, Allfaces, Allords, AllCharges = deleteBase(AllRefs, Allfaces, Allords, AllCharges, x_0, y_0)
#Same plotting, but now different color for different sign of charges
figure, ax = plt.subplots(1)
ax.scatter(x_0, y_0)
AllXs = column(AllRefs, 0)
AllYs = column(AllRefs, 1)
posXs = [b for a, b in zip(AllCharges, AllXs) if a]
posYs = [b for a, b in zip(AllCharges, AllYs) if a]
negXs = [b for a, b in zip(AllCharges, AllXs) if not a]
negYs = [b for a, b in zip(AllCharges, AllYs) if not a]
ax.scatter(posXs, posYs)
ax.scatter(negXs, negYs)
rect = patch.Rectangle((-x_box, -y_box), 2*x_box, 2*y_box, fill=False, linewidth=2.5)
ax.add_patch(rect)
plt.show()

xmirror = column(AllRefs,0)
xc = [heatx]
xc.extend(xmirror)
ymirror = column(AllRefs,1)
yc = [heaty]
yc.extend(ymirror)

signmirror = np.zeros(len(xmirror))
for i in range(0,len(xmirror)):
    if AllCharges[i] == True:
        signmirror[i] = 1
    if AllCharges[i] == False:
        signmirror[i] = -1
        
sign = [heat_sign]
sign.extend(signmirror)

xc = xc/np.sqrt(Dx)
yc = yc/np.sqrt(Dy)
zc = heatz*np.ones(len(xc))
xc = np.append(xc, xc)
yc = np.append(yc, yc)
zc = np.append(zc, -3*zc)
oppsign = [i * -1 for i in sign]
sign = np.append(sign, oppsign)
    

#Finished finding mirror points

density = (xpoints*ypoints*zpoints)/(truex*truey*truez)

p0 = Point (-xmax/2, -ymax/2, -zmax/2)
p1 = Point (xmax/2, ymax/2, zmax/2)
mesh = BoxMesh(p0, p1, xpoints, ypoints, zpoints)

CG1_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
CG2_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem = MixedElement([CG1_elem, CG2_elem])
W = FunctionSpace(mesh, W_elem)
  
kx = ktrue*Dx
ky = ktrue*Dy

#Define the expression for the beam used to solve the differential equation (ignore mirror charges)
if heat_sign == 1:
    g = Expression('exp(b*(x[2] - top/2) - kx*(x[0]-xcenter)*(x[0]-xcenter) - ky*(x[1]-ycenter)*(x[1]-ycenter))', degree = 2, kx = kx, ky = ky, b = b*np.sqrt(Dz), top = zmax/2, xcenter = heatx, ycenter = heaty)
if heat_sign == -1:
    g = Expression('- exp(b*(x[2] - top/2) - kx*(x[0]-xcenter)*(x[0]-xcenter) - ky*(x[1]-ycenter)*(x[1]-ycenter))', degree = 2, kx = kx, ky = ky, b = b*np.sqrt(Dz), top = zmax/2, xcenter = heatx, ycenter = heaty)

urbase, uibase = solver(g, xmax, ymax, zmax, tol, mesh, omega, alpha1, 0, Dx, Dy)        
ureal1, uimag1 = solver(g, xmax, ymax, zmax, tol, mesh, omega, alpha1, Dxy1, Dx, Dy)

meshpts = mesh.coordinates()
    
TrBase, TiBase = Temp_all_mesh(xc, yc, zc, omega, A, mesh, sign)
Di, Dr = DxyBoundaryTerms(meshpts[:, 0], meshpts[:, 1], meshpts[:, 2], heatx, heaty, heatz, x_box, y_box, heatz, Dx, Dy, Dz, Dxy1, omega, A, sp)
Ti = TiBase + Di
Tr = TrBase + Dr
Tphase = np.arctan2(Ti, Tr)
TphaseBase = np.arctan2(TiBase, TrBase)
#Tphase = np.arctan2(Ti + Di, Tr + Dr)

cutnumy = xpoints+1
cutnumx = ypoints+1
cutnumz = zpoints+1

xn = -truex/2
xp = truex/2
zn = -truez/2
zp = truez/2
yp = truey/2
yn = -truey/2

y = np.linspace(yn + tol, yp - tol, cutnumy)
x = np.linspace(xn + tol, xp - tol, cutnumx)
z = np.linspace(zn + tol, zp - tol, cutnumz)

utotr1, utoti1 = utotal(cutnumx, cutnumy, x, y, z, cutnumz, ureal1, uimag1, depth)
utotrBase, utotiBase = utotal(cutnumx, cutnumy, x, y, z, cutnumz, urbase, uibase, depth)

TrBase2d, TiBase2d = Ttotal(cutnumx, cutnumy, x, y, z, cutnumz, xc, yc, zc, depth, sign)
Dr2d, Di2d = DxyTotal(cutnumx, cutnumy, x, y, z, cutnumz, heatx, heaty, heatz, x_box, y_box, heatz, depth, omega, Dx, Dy, Dz, Dxy1, A, sp)
Tr2d = TrBase2d + Dr2d
Ti2d = TiBase2d + Di2d

tanSim = np.arctan2(utoti1, utotr1)
tanBaseSim = np.arctan2(utotiBase, utotrBase)

tanIdeal = np.arctan2(Ti2d, Tr2d)
tanBaseIdeal = np.arctan2(TiBase2d, TrBase2d)

pts = linepoints(np.pi/2, .42, 0, truex, truey, truez, spacing)
ds = pts[:,0]
LineXs = pts[:,1]
LineYs = pts[:,2]
LineIdealR, LineIdealI = lineTemp(LineXs, LineYs, z, cutnumz, heatx, heaty, heatz, xc, yc, zc, sign, heat_sign, x_box, y_box, heatz, depth, omega, Dx, Dy, Dz, Dxy1, A, sp)
LineBaseIdealR, LineBaseIdealI = lineTemp(LineXs, LineYs, z, cutnumz, heatx, heaty, heatz, xc, yc, zc, sign, heat_sign, x_box, y_box, heatz, depth, omega, Dx, Dy, Dz, 0, A, sp)
PhaseLineSim = phaseline(ureal1, uimag1, LineXs, LineYs, depth, z)
PhaseLineBase = phaseline(urbase, uibase, LineXs, LineYs, depth, z)

def dO(f):
    lmax = max(f)
    n = np.abs(lmax)//(np.pi*2)*np.sign(lmax)
    return f - 2*np.pi*n

LineUnwrapIdeal = np.unwrap(np.arctan2(LineIdealI, LineIdealR))
LineUnwrapIdealBase =  np.unwrap(np.arctan2(LineBaseIdealI, LineBaseIdealR))
LineUnwrapIdeal = dO(LineUnwrapIdeal)
LineUnwrapIdealBase = dO(LineUnwrapIdealBase)

LineUnwrapSim = dO(np.unwrap(PhaseLineSim))
LineUnwrapSimBase = dO(np.unwrap(PhaseLineBase))

print('Phase Plots')

xv, yv = np.meshgrid(x, y)

ideal_unwrapped = unwrap_phase(np.transpose(tanIdeal))
base_ideal_uw = unwrap_phase(np.transpose(tanBaseIdeal))

ideal_unwrapped = delOffset(ideal_unwrapped)
base_ideal_uw = delOffset(base_ideal_uw)

sim_unwrapped = unwrap_phase(np.transpose(tanSim))
base_sim_uw = unwrap_phase(np.transpose(tanBaseSim))

sim_unwrapped = delOffset(sim_unwrapped)
base_sim_uw = delOffset(base_sim_uw)

fig = plt.gcf()
plt.contourf(xv, yv, ideal_unwrapped, 100, cmap = cmx.jet)
plt.plot(LineXs, LineYs, c = 'black', linewidth = 3)
plt.colorbar()
plt.title('Ideal Unwrapped Phase')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

fig = plt.gcf()
plt.contourf(xv, yv, sim_unwrapped, 100, cmap = cmx.jet)
plt.plot(LineXs, LineYs, c = 'black', linewidth = 3)
plt.colorbar()
plt.title('Simulated Unwrapped Phase')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("phase.pdf", bbox_inches='tight')
plt.show()

sim_signal = sim_unwrapped - base_sim_uw
ideal_signal = ideal_unwrapped - base_ideal_uw

fig = plt.gcf()
plt.contourf(xv, yv, sim_signal, 100, cmap = cmx.jet)
plt.plot(LineXs, LineYs, c = 'black', linewidth = 3)
plt.colorbar()
plt.title('Simulated Dxy Signal')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("DxySimSig.pdf", bbox_inches='tight')
plt.show()

fig = plt.gcf()
plt.contourf(xv, yv, ideal_signal, 100, cmap = cmx.jet)
plt.plot(LineXs, LineYs, c = 'black', linewidth = 3)
plt.colorbar()
plt.title('Ideal Dxy Signal')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("DxyIdealSig.pdf", bbox_inches='tight')
plt.show()

fig = plt.gcf()
plt.contourf(xv, yv, sim_signal - ideal_signal, 100, cmap = cmx.jet)
plt.plot(LineXs, LineYs, c = 'black', linewidth = 3)
plt.colorbar()
plt.title('Dxy Model Error')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("DxyError.pdf", bbox_inches='tight')
plt.show()

plt.plot(LineUnwrapIdeal)
plt.plot(LineUnwrapIdealBase)
plt.plot(LineUnwrapSim)
plt.plot(LineUnwrapSimBase)
plt.show()

plt.plot(LineYs, LineUnwrapIdeal - LineUnwrapIdealBase, label='Ideal Signal')
plt.plot(LineYs, LineUnwrapSim - LineUnwrapSimBase, label='Sim Signal')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from functions import *

filename = './mhd_sheet/fmhd_b_000001.100000'
open_file_fmhd(filename)

Bx = Load_file_3D("./fields/Field in x")
By = Load_file_3D("./fields/Field in y")
Bz = Load_file_3D("./fields/Field in z")

iterations=1000
approx_method = MultivariableRungaKutta(Bx,By,Bz,iterations)
n=3
points=3
li=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
xi=[[0.254,0.253,0.25,0.255,0.253,0.25,0.249,0.248,0.254,0.251],[0.732,0.736,0.73,0.730,0.734,0.733,0.731,0.732,0.733,0.735],[0.453,0.454,0.456,0.452,0.453,0.454,0.455,0.456,0.457,0.458]]
yi=[[0.732,0.736,0.73,0.730,0.734,0.733,0.731,0.732,0.733,0.735],[0.453,0.454,0.456,0.452,0.453,0.454,0.455,0.456,0.457,0.458],[0.254,0.253,0.25,0.255,0.253,0.25,0.249,0.248,0.254,0.251]]
zi=[[0.453,0.454,0.456,0.452,0.453,0.454,0.455,0.456,0.457,0.458],[0.254,0.253,0.25,0.255,0.253,0.25,0.249,0.248,0.254,0.251],[0.732,0.736,0.73,0.730,0.734,0.733,0.731,0.732,0.733,0.735]]
hi=[1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7]

l=[[0 for col in range(iterations)] for row in range(n*points)]
x=[[0 for col in range(iterations)] for row in range(n*points)]
y=[[0 for col in range(iterations)] for row in range(n*points)]
z=[[0 for col in range(iterations)] for row in range(n*points)]
H=[[0 for col in range(iterations)] for row in range(n*points)]
iteration=[[0 for col in range(iterations)] for row in range(n*points)]

k=0
for i in range(points):
    for j in range(n):
        l[k],x[k],y[k],z[k],H[k],iteration[k] = approx_method.variable_step_size_approximation(li[j],xi[i][j],yi[i][j],zi[i][j],hi[j],interpolation = True)
        k+=1

del_l=list()
h=list()
for i in range(k):   
    h.append((np.mean(H[i])))
    del_l.append(l[i][-1])
l_c=np.arange(0.,np.min(del_l),np.mean(h))
length=len(l_c)

rl=[[l_c] for row in range(n)]
rx=[[0 for col in range(length)] for row in range(n*points)]
ry=[[0 for col in range(length)] for row in range(n*points)]
rz=[[0 for col in range(length)] for row in range(n*points)]
rH=[[h for col in range(length)] for row in range(n*points)]

for i in range(k):
    rx[i]=np.interp(l_c,l[i],x[i])
    ry[i]=np.interp(l_c,l[i],y[i])
    rz[i]=np.interp(l_c,l[i],z[i])


#Evaluates and Plots the  Richardson diffusion of the field lines 
r_diff=[]
for p in range(len(l_c)):
    diff=[]
    for i in range(points):
        for q in range(n-1):
            for o in range(q+1,n):
                try:
                    diff.append(np.sqrt((rx[i*n+q][p]-rx[i*n+o][p])**2+(ry[i*n+q][p]-ry[i*n+o][p])**2+(rz[i*n+q][p]-rz[i*n+o][p])**2))
                except:
                    break
    r_diff.append(np.mean(diff))

ax=plt.axes()
plt.xscale("log")
plt.yscale("log")
ax.set_xlabel("l")
ax.set_ylabel("Diffusion")
ax.set_title("Richardson Diffusion") 
ax.plot(l_c,r_diff)
plt.show()
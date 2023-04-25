from functions import *
import matplotlib.pyplot as plt

filename = './mhd_sheet/fmhd_b_000001.100000'
open_file_fmhd(filename)

Bx = Load_file_3D("./fields/Field in x")
By = Load_file_3D("./fields/Field in y")
Bz = Load_file_3D("./fields/Field in z")

iterations=1000
approx_method = MultivariableRungaKutta(Bx,By,Bz,iterations)
n=3
l=[[0 for col in range(iterations)] for row in range(n)]
x=[[0 for col in range(iterations)] for row in range(n)]
y=[[0 for col in range(iterations)] for row in range(n)]
z=[[0 for col in range(iterations)] for row in range(n)]
H=[[0 for col in range(iterations)] for row in range(n)]
iteration=[[0 for col in range(iterations)] for row in range(n)]
li=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
xi=[0.25,0.25,0.25,0.255,0.253,0.25,0.249,0.248,0.254,0.251]
yi=[0.73,0.73,0.73,0.730,0.734,0.733,0.731,0.732,0.733,0.735]
zi=[0.2,0.4,0.6,0.452,0.453,0.454,0.455,0.456,0.457,0.458]
hi=[1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7]

for j in range(n):
    l[j],x[j],y[j],z[j],H[j],iteration[j] = approx_method.variable_step_size_approximation(li[j],xi[j],yi[j],zi[j],hi[j],interpolation = True)

fig = plt.figure()
ax = plt.axes(projection='3d')
for j in range(n):
    ax.plot3D(np.array(x[j]), np.array(y[j]), np.array(z[j]))
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis ")
ax.set_title("Field lines of " + filename[-20:])
plt.show()

Save_file_3D(np.asanyarray(l,dtype=object),"./fields/field_lines_l")
Save_file_3D(np.asanyarray(x,dtype=object),"./fields/field_lines_x")
Save_file_3D(np.asanyarray(y,dtype=object),"./fields/field_lines_y")
Save_file_3D(np.asanyarray(z,dtype=object),"./fields/field_lines_z")
Save_file_3D(np.asanyarray(H,dtype=object),"./fields/field_lines_H")






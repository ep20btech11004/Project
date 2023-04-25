from file_fn import Load_file_3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[-2],ys[-2]),(xs[-1],ys[-1]))
        return np.min(zs)

l = Load_file_3D("./fields/field_lines_l")
x = Load_file_3D("./fields/field_lines_x")
y = Load_file_3D("./fields/field_lines_y")
z = Load_file_3D("./fields/field_lines_z")
H = Load_file_3D("./fields/field_lines_H")
n=3
fig = plt.figure()
ax = plt.axes(projection='3d')
for j in range(n):
    ax.plot3D(np.array(x[j]), np.array(y[j]), np.array(z[j]))
    a = Arrow3D(x[j], y[j], z[j], mutation_scale=15, lw=1, arrowstyle="-|>", color="black")
    ax.add_artist(a)
ax.set_title("Plot of Field Lines")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis ")
plt.show()

import numpy as np
import scipy.optimize

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas



fig = plt.figure()
ax = fig.gca(projection='3d')
def fitPlaneLTSQ(XYZ):
    (rows, cols) = XYZ.shape
    print(rows, cols)

    G = np.ones((rows, 3))
    print(XYZ)
    print(XYZ[0])
    G[:, 0] = XYZ[0]  # X
    G[:, 1] = XYZ[1]  # Y
    print(G)

    Z = XYZ[2]

    (a, b, c), resid, rank, s = np.linalg.lstsq(G, Z)
    print(a , b , c, resid, rank, s)

    normal = (a, b, -1)
    nn = np.linalg.norm(normal)
    normal = normal/nn

    print(normal)

    return (c, normal)

Filename = "xxx"
#Import Data from CSV
result = pandas.read_csv(Filename, header=None) # , names=['X', 'Y','Z']
#result =result.head(5)
print(result)




#standard normal distribution / Bell.
#np.random.seed(seed=1)

data = result
#print(data)
print("NEW : ")
print(data)
c, normal = fitPlaneLTSQ(data)
print(c, normal)

# plot fitted plane
maxx = np.max(data[0])
maxy = np.max(data[1])
minx = np.min(data[0])
miny = np.min(data[1])
print(maxx,maxy, minx, miny)

point = np.array([0.0, 0.0, c])
print(point)
d = -point.dot(normal)
print(d)

# plot original points
ax.scatter(data[0], data[1], data[2])

# compute needed points for plane plotting
xx, yy = np.meshgrid([minx, maxx], [miny, maxy])
z = (-normal[0]*xx - normal[1]*yy - d)*1. / normal[2]

# plot plane
ax.plot_surface(xx, yy, z, alpha=0.2)
ax.set_xlim(-1, 1)
ax.set_ylim(-1,1)
ax.set_zlim(1,2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

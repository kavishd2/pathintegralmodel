import sympy as sym
import math
import matplotlib.pyplot as plt
import numpy as np

#Define symbols
t = sym.Symbol('t')
x = sym.Symbol('x')
y = sym.Symbol('y')
z = sym.Symbol('z')

#Define the vector field F with components f, g, h
f = sym.sin(z)
g = sym.cos(z)
h = (x*y)**(1/3)
print(f"F=<{f},{g},{h}>")

#Define path parametrization x(t), y(t), z(t)
x = (sym.cos(t))**3
y = (sym.sin(t))**3
z = t

#Compute derivatives x'(t), y'(t), z'(t)
x_prime = x.diff(t)
y_prime = y.diff(t)
z_prime = z.diff(t)
f = sym.sin(z)
g = sym.cos(z)
h = (x*y)**(1/3)

#Compute the dot product
dot_product = f*x_prime+g*y_prime+h*z_prime

#Compute the path integral
path_integral = sym.integrate(dot_product, (t, 0, 7*math.pi/2))
print(f"r=<{x},{y},{z}>")
print("Path Integral:", path_integral)

#Plotting the vector field
fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection='3d')
ax.grid()

a, b, c = np.meshgrid(np.arange(-1, 1.5, .5),
                      np.arange(-1, 1.5, .5),
                      np.arange(-0.5, 12, 1))

u = np.sin(c)
v = np.cos(c)
w = np.sign(a*b) * (np.abs(a*b)) ** (1 / 3)

ax.quiver(a, b, c, u, v, w, length=0.1, color = 'black')

#Plotting the path
s = np.arange(0, 7*np.pi/2, np.pi/1000)
d = (np.cos(s))**3
e = (np.sin(s))**3
f = s
ax.plot3D(d, e, f)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

plt.show()

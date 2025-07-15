import numpy as np
import matplotlib.pyplot as plt

L = 10.0
N = 100
psi = np.loadtxt("/tmp/psi1.csv").reshape(N, N)
plt.imshow(psi, extent=[-L, L, -L, L], origin='lower', cmap='viridis')
plt.colorbar(label='ψ')
plt.title('Wavefunction Heatmap')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)
Z = psi  # or V for potential

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('ψ')
plt.title('Wavefunction 3D Surface')
plt.show()

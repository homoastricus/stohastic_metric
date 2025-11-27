import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ФИКС ДЛЯ PyCharm
import matplotlib
matplotlib.use('Qt5Agg')
plt.ion()

# 32 ТОЧКИ
dx_range = np.arange(-2, 3)
dy_range = np.arange(-2, 3)
dz_range = np.arange(-2, 3)
X, Y, Z = np.meshgrid(dx_range, dy_range, dz_range, indexing='ij')
distances = np.sqrt(X**2 + Y**2 + Z**2)
mask = (distances <= 2.0) & (distances > 0)
points_32 = np.column_stack([X[mask], Y[mask], Z[mask]])

# ГЛАВНОЕ ОКНО
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# ТОЧКИ
ax.scatter(0, 0, 0, c='gold', s=500, marker='*', label='Центр')
ax.scatter(points_32[:,0], points_32[:,1], points_32[:,2],
           c='red', s=300, marker='o', label='32 точки')

# ЛИНИИ
for point in points_32:
    ax.plot([0, point[0]], [0, point[1]], [0, point[2]], 'k-', alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('🖱️ 32 ТОЧКИ КОРРЕЛЯЦИИ')
ax.legend()

# ✅ ФИНАЛЬНЫЙ ФИКС
plt.tight_layout()
plt.show(block=True)
print("🖱️ ЛКМ + ДВИЖЕНИЕ = ВРАЩАНИЕ!")
print("🖱️ КОЛЁСИКО = ЗУМ")
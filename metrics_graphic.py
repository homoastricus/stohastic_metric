import numpy as np
import matplotlib.pyplot as plt

# Параметры из модели
std_alpha_grid600 = 0.000475   # std(α) для grid_size = 600
N_grid600 = 600**3             # 216,000,000 ячеек
N_universe = 1e185             # Вселенная

# Формируем ряд N от маленьких сеток до Вселенной
N_values = np.logspace(6, 185, num=100, base=10)  # от 1e6 до 1e185

# Закон больших чисел: std(α) ~ 1/sqrt(N)
std_values = std_alpha_grid600 * np.sqrt(N_grid600 / N_values)

# Визуализация
plt.figure(figsize=(8,6))
plt.loglog(N_values, std_values, 'b-', lw=2, label='σ(α) по закону больших чисел')
plt.scatter([N_grid600, N_universe],
            [std_alpha_grid600, std_alpha_grid600*np.sqrt(N_grid600/N_universe)],
            color='red', label='Сетка / Вселенная')
plt.xlabel('Число ячеек N')
plt.ylabel('σ(α)')
plt.title('Предел стохастических флуктуаций α vs число ячеек')
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.legend()
plt.show()

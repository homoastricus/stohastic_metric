import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 20. ВИЗУАЛИЗАЦИЯ МАСШТАБНОГО ПЕРЕХОДА
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("ВИЗУАЛИЗАЦИЯ: TOY-МОДЕЛЬ → РЕАЛЬНЫЙ МИР")
print("=" * 70)


def scale_to_real_world(toy_values, scale_factor=1.616e-35):
    """Масштабирование величин из toy-модели в реальный мир"""

    real_world = {}

    # Пространственные масштабы
    real_world['planck_length'] = toy_values['planck_length'] * scale_factor
    real_world['universe_size'] = toy_values['universe_size'] * scale_factor
    real_world['correlation_length'] = toy_values['correlation_length'] * scale_factor

    # Безразмерные величины (инварианты)
    real_world['strong_correlations'] = toy_values['strong_correlations']  # 32
    real_world['holographic_ratio'] = toy_values['holographic_ratio']  # 1.5

    # Статистические величины (масштабируются)
    real_world['correlated_fraction'] = toy_values['correlated_fraction'] * scale_factor ** 3
    real_world['effective_clusters'] = toy_values['effective_clusters'] / scale_factor ** 3
    real_world['macro_fluctuations'] = toy_values['macro_fluctuations'] * scale_factor

    return real_world


# Данные из вашей toy-модели
toy_model = {
    'planck_length': 1.0,
    'universe_size': 450.0,
    'correlation_length': 2.0,
    'strong_correlations': 32,
    'holographic_ratio': 1.5,
    'correlated_fraction': 3.5e-7,  # 0.000035%
    'effective_clusters': 2.8e6,
    'macro_fluctuations': 0.01
}

real_world = scale_to_real_world(toy_model)

print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА:")
print("Параметр              | Toy-модель     | Реальный мир      | Изменение")
print("-" * 75)

params = [
    ("Планковская длина", "1.0", f"{real_world['planck_length']:.1e} м", f"× {1.616e-35:.1e}"),
    ("Размер системы", "450", f"{real_world['universe_size']:.1e} м", f"× {1.616e-35:.1e}"),
    ("Коррелир. ячеек", "32", "32", "инвариант"),
    ("Доля коррелир.", f"{toy_model['correlated_fraction']:.1e}", f"{real_world['correlated_fraction']:.1e}",
     f"× {1.616e-35 ** 3:.1e}"),
    ("Число кластеров", f"{toy_model['effective_clusters']:.1e}", f"{real_world['effective_clusters']:.1e}",
     f"× {1.616e-35 ** -3:.1e}"),
    ("Макро σ(α)", f"{toy_model['macro_fluctuations']:.2f}", f"{real_world['macro_fluctuations']:.1e}",
     f"× {1.616e-35:.1e}"),
]

for name, toy_val, real_val, change in params:
    print(f"{name:<20} | {toy_val:<14} | {real_val:<17} | {change}")

# ---------------------------------------------------------
# 21. ГРАФИКИ МАСШТАБНОГО ПЕРЕХОДА
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# График 1: Изменение доли коррелирующих ячеек
scales = np.logspace(0, -35, 100)  # от toy-модели к реальному миру
correlated_fractions = toy_model['correlated_fraction'] * scales ** 3

axes[0, 0].semilogy(-np.log10(scales), correlated_fractions, 'r-', linewidth=3)
axes[0, 0].axvline(x=35, color='black', linestyle='--', label='Реальный мир')
axes[0, 0].set_xlabel('-log₁₀(масштабный коэффициент)')
axes[0, 0].set_ylabel('Доля коррелирующих ячеек')
axes[0, 0].set_title('Экстремальное уменьшение доли\nкоррелирующих ячеек')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# График 2: Рост числа независимых кластеров
effective_clusters = toy_model['effective_clusters'] / scales ** 3

axes[0, 1].semilogy(-np.log10(scales), effective_clusters, 'b-', linewidth=3)
axes[0, 1].axvline(x=35, color='black', linestyle='--', label='Реальный мир')
axes[0, 1].set_xlabel('-log₁₀(масштабный коэффициент)')
axes[0, 1].set_ylabel('Эффективное число кластеров')
axes[0, 1].set_title('Экспоненциальный рост числа\nнезависимых корреляционных кластеров')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# График 3: Уменьшение макроскопических флуктуаций
macro_fluctuations = toy_model['macro_fluctuations'] * scales

axes[0, 2].semilogy(-np.log10(scales), macro_fluctuations, 'g-', linewidth=3)
axes[0, 2].axvline(x=35, color='black', linestyle='--', label='Реальный мир')
axes[0, 2].set_xlabel('-log₁₀(масштабный коэффициент)')
axes[0, 2].set_ylabel('Макроскопические σ(α)')
axes[0, 2].set_title('Резкое уменьшение макроскопических\nфлуктуаций метрики')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# График 4: Голографическое соотношение (инвариант!)
holographic_ratio = np.full_like(scales, toy_model['holographic_ratio'])

axes[1, 0].plot(-np.log10(scales), holographic_ratio, 'purple', linewidth=3)
axes[1, 0].axvline(x=35, color='black', linestyle='--', label='Реальный мир')
axes[1, 0].set_xlabel('-log₁₀(масштабный коэффициент)')
axes[1, 0].set_ylabel('Соотношение поверхность/объем')
axes[1, 0].set_title('Инвариантность голографического\nсоотношения')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# График 5: Число сильно коррелирующих ячеек (инвариант!)
strong_correlations = np.full_like(scales, toy_model['strong_correlations'])

axes[1, 1].plot(-np.log10(scales), strong_correlations, 'orange', linewidth=3)
axes[1, 1].axvline(x=35, color='black', linestyle='--', label='Реальный мир')
axes[1, 1].set_xlabel('-log₁₀(масштабный коэффициент)')
axes[1, 1].set_ylabel('Сильно коррелирующих ячеек')
axes[1, 1].set_title('Абсолютная инвариантность числа\nсильно коррелирующих ячеек')
axes[1, 1].legend()
axes[1, 0].grid(True, alpha=0.3)

# График 6: Сравнение законов силы
r_toy = np.linspace(1, 100, 100)
scale_factor = 1.616e-35
r_real = r_toy * scale_factor

# В toy-модели переход медленнее
alpha_toy = 2.0 + 0.5 * np.exp(-r_toy / 20)
# В реальном мире переход почти мгновенный
alpha_real = 2.0 + 0.5 * np.exp(-r_real / (2 * 1.616e-35))

axes[1, 2].semilogx(r_toy, alpha_toy, 'red', label='Toy-модель', linewidth=2)
axes[1, 2].semilogx(r_toy, alpha_real, 'blue', label='Реальный мир', linewidth=2)
axes[1, 2].axhline(y=2.0, color='black', linestyle='--', label='Классический предел')
axes[1, 2].set_xlabel('Расстояние (в единицах toy-модели)')
axes[1, 2].set_ylabel('α(r)')
axes[1, 2].set_title('Ускоренный переход к классическому\nпределу в реальном мире')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
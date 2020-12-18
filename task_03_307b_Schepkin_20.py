import math
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

# Функция гауссового импульса
def gauss(q, m, d_g, w_g, d_t):
    return np.exp(-((((q - m) - (d_g / d_t))
                     / (w_g / dt)) ** 2))

# Параметры моделирования
# Волновое сопротивление свободного пространства
W0 = 120.0 * np.pi

# Число Куранта
Sc = 1.0

# Скорость света
c = 299792458.0

# Размер области моделирования в метрах
maxSize_m = 1.8
# Дискрет по пространству
dx = 5e-3
# Размер облести моделирования в отсчетах
maxSize = math.floor(maxSize_m / dx + 0.5)

# Время расчета в отсчетах
maxTime = 512
# Дискрет по времени
dt = Sc * dx / c
# временна сетка
tlist = np.arange(0, maxTime * dt, dt)

# Шаг по частоте
df = 1.0 / (maxTime * dt)
# Частотная сетка
freq = np.arange(-maxTime / 2 * df, maxTime / 2 * df, df)

# Уровень ослабления в начальный момент времени
A_0 = 100
# Уровень ослабления на частоте F_max
A_max = 100
# Ширина спектра по уровню 0.01
F_max = 2.5e9
# Параметр "ширины" импульса
wg = np.sqrt(np.log(A_max)) / (np.pi * F_max)
# Параметр "задержки" импульса
dg = wg * np.sqrt(np.log(A_0))

# Положение источника в метрах
sourcePos_m = 0.2
# Положение источника в отсчетах
sourcePos = math.floor(sourcePos_m / dx + 0.5)

# Положение датчика в метрах
probePos_m = 1
# Положение датчика в отсчетах
probePos = math.floor(probePos_m / dx + 0.5)

# Инициализация датчика
probeEz = np.zeros(maxTime)
probeHy = np.zeros(maxTime)

# Инициализация полей
Ez = np.zeros(maxSize)
Hy = np.zeros(maxSize)

# Пространственная сетка
xlist = np.arange(0, maxSize_m, dx)

# Включение интерактивного режима для анимации
plt.ion()
# Создание окна для графика
fig, ax = plt.subplots()

# Установка отображаемых интервалов по осям
ax.set_xlim(0, maxSize_m)
ax.set_ylim(-0.6, 1.2)

# Установка меток по осям
ax.set_xlabel('x, м')
ax.set_ylabel('Ez, В/м')

# Включение сетки на графике
ax.grid()

# Отображение источника и датчика
ax.plot(sourcePos_m, 0, 'ok')
ax.plot(probePos_m, 0, 'xr')

# Отображение поля в начальный момент
line, = ax.plot(xlist, Ez)

# Рассчет полей
for t in range(1, maxTime):
    # Граничные условия для поля H
    Hy[-1] = Hy[-2]
    # Расчет компоненты поля H
    Hy[:-1] = Hy[:-1] + (Ez[1:] - Ez[:-1]) * Sc / W0
    # Источник возбуждения с использованием метода
    # Total Field / Scattered Field
    Hy[sourcePos - 1] -= (Sc / W0) * gauss(t, sourcePos, dg, wg, dt)

    # Граничные условия для поля E
    Ez[0] = Ez[1]
    # Расчет компоненты поля E
    Ez[1:] = Ez[1:] + (Hy[1:] - Hy[:-1]) * Sc * W0
    # Источник возбуждения с использованием метода
    # Total Field / Scattered Field
    Ez[sourcePos] += Sc * gauss(t + 1, sourcePos, dg, wg, dt)

    # Регистрация поля в датчиках
    probeHy[t] = Hy[probePos]
    probeEz[t] = Ez[probePos]

    # Обновление графика
    if t % 4 == 0:
        plt.title(format(t * dt * 1e9, '.3f') + ' нc')
        line.set_ydata(Ez)
        fig.canvas.draw()
        fig.canvas.flush_events()

# Выключение интерактивного режима
plt.ioff()

# Cпектр сигнала
EzSpec = fftshift(np.abs(fft(probeEz)))

# Вывод сигнала и спектра
fig, (ax1, ax2) = plt.subplots(2, 1)
# Сигнал
ax1.set_xlim(0, maxTime * dt / 1.5)
ax1.set_ylim(0, 1.2)
ax1.set_xlabel('t, нс')
ax1.set_ylabel('Ez, В/м')
ax1.plot(tlist, probeEz)
ax1.minorticks_on()
ax1.grid()
# Спектр
ax2.set_xlim(0, maxTime * df / 20)
ax2.set_ylim(0, 1.2)
ax2.set_xlabel('f, ГГц')
ax2.set_ylabel('|S| / |Smax|, б/р')
ax2.plot(freq, EzSpec / np.max(EzSpec))
ax2.minorticks_on()
ax2.grid()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import os
n = 100
A = -3.13024
e = 2.718281828
y = np.linspace(-100, 100, 100)
x = np.linspace(-2*np.pi, 2*np.pi, 100) # интервал
fx = (np.sin(y)*(e**((1 - np.cos(A))**(2))) + (np.cos(A)*(e**((1 - np.sin(A))**2)))+(x - A)**(2)) 
# создание папки для результатов
try:
 os.mkdir('')
except OSError:
 pass
complete_file = os.path.join('results', 'task_01_307b_Schepkin_20.txt')
f = open(complete_file, 'w')
#txt файл с результатами
f.write('   x    f(x)\n')
for i in range(n):
 f.write(str("%.4f" % x[i])+' '+str("%.4f" % fx[i])+"\n")
f.close()
# построение графика
fig, ax = plt.subplots()
ax.plot(x, fx, linewidth = 2)
ax.set_xlim(-2*np.pi, 2*np.pi);
ax.set_ylim(-60, 140);
ax.grid(linewidth = 1)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, fx, color = 'green')
plt.show()

import numpy as np

light = np.zeros(101)
for i in range(1, 101):
    for j in range(i, 101):
        if j % i == 0:
            light[j] = not light[j]

print(sum(light))

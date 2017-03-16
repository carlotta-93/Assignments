import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt('diatoms.txt')


third_diatoms_coordinates = data[2]
td_x = third_diatoms_coordinates[::2]
td_y = third_diatoms_coordinates[1::2]

all_x = []
all_y = []
for diatoms in data:
    all_x.append(diatoms[::2])
    all_y.append(diatoms[1::2])

all_diatoms = plt.scatter(all_x, all_y, marker='o', c='#9999ff', label='all diatoms', s=10, facecolor='0.5', lw=0.5)
third_diatoms = plt.scatter(td_x, td_y, marker='o', c='orange', label='third diatoms', s=10, facecolor='0.5', lw=0.5)
plt.axis('equal')
plt.xlabel('x coordinates')
plt.ylabel('y coordinates')
plt.title('Shape of all diatoms')
plt.legend(loc='lower right')
plt.show(all_diatoms)


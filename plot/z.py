import matplotlib.pyplot as plt
import numpy as np

x_data = np.arange(1, 11, 1)
y1_data = [29.6812, 21.6829, 14.1905, 7.55355, 1.65948, 0, 0, 0, 0, 0]
y2_data = [29.6812, 21.6829, 14.1905, 8.26556, 3.94441, 1.1891, 0, 0, 0, 0]
y3_data = [44.840625, 31.522687, 21.280922, 13.353274, 7.316957, 3.124488, 1.184094, 0.415695, 0.030042, 0.000000]
y4_data = [4, 5]

fig = plt.figure()

left, bottom, width, height = 0.13, 0.2, 0.8, 0.7
ax = fig.add_axes([left, bottom, width, height])

# ax = fig.add_subplot(111)
ax.set(xlim=[0.9, 10.1], ylim=[-0.3, 45.1], title='')
plt.xlabel('The number of blocks between the target TX \nand the confirmation (represented by $z$)',
           fontdict={'family': 'Times New Roman', 'size': 18})
plt.ylabel('Expected reward of the attacker', fontdict={'family': 'Times New Roman', 'size': 18})
plt.yticks(fontproperties='Times New Roman', size=16)
plt.xticks(fontproperties='Times New Roman', size=16)

plt.plot(x_data, y1_data, color='red', linewidth=1.5, marker='^', markersize=6, label='Naive DSA', linestyle='-')
plt.plot(x_data, y2_data, color='green', linewidth=1.5, marker='s', markersize=6, label='Adaptive DSA', linestyle='--')
plt.plot(x_data, y3_data, color='blue', linewidth=1.5, marker='o', markersize=6, label='RA-DSA $w = 0.5$',
         linestyle='-.')
# plt.plot(x_data,y4_data,color='c',linewidth=1.5,marker='*',markersize=6,label='w=0.75',linestyle=':')

ax.legend()
plt.legend(prop={'family': 'Times New Roman', 'size': 18})

ax1 = fig.add_axes([0.6, 0.31, 0.32, 0.3])  # 不用figure的形式则无须用set
y4 = [0.415695, 0.030042, 0.000000]
y5 = [0, 0, 0]
x = [8, 9, 10]

plt.tick_params(labelsize=16)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax1.plot(x, y4, color='blue', linewidth=1.5, marker='o', markersize=6, label='RA-DSA $w = 0.5$', linestyle='-.')
ax1.plot(x, y5, color='green', linewidth=1.5, marker='s', markersize=6, label='Adaptive DSA', linestyle='--')
ax1.plot(x, y5, color='red', linewidth=1.5, marker='^', markersize=6, label='Naive DSA', linestyle='-')

plt.savefig('z.svg')
plt.show()

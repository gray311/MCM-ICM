import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 5, 30)
y1 = x ** 2 + x * 2 + 1
y2 = x * 2 + x + 1


plt.figure()

l1,= plt.plot(x, y1)

l2,= plt.plot(x, y2,color='red',linewidth=2.0,linestyle='--')

plt.xlim((-1, 2))
plt.ylim((-2, 5))

#set new name
plt.xlabel('I am x')
plt.ylabel('I am y')

# set new sticks

new_ticks = np.linspace(-1, 2, 7)
print(new_ticks)
plt.xticks(new_ticks)

# set tick labels

plt.yticks([-2, -1.8, -1, 1.22, 3], [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])

# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

#注释
plt.legend(handles=[l1, l2], labels=['up', 'down'], loc='best')

#标注点
x0 = 1
y0 = x0 ** 2 + x0 * 2 + 1
plt.plot([x0, x0,], [0, y0,], 'k--', linewidth=2)
plt.scatter([x0, ], [y0, ], s=50, color='b')

# method 1:
#####################
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30),textcoords='offset points', fontsize=16,arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

# method 2:
########################
plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$', fontdict={'size': 16, 'color': 'r'})

#ticks vision
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(12)
    # set zorder for ordering the plot in plt 2.0.2 or higher
    label.set_bbox(dict(facecolor='red', edgecolor='none', alpha=0.7, zorder=2))

plt.show()
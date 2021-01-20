import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 5, 30)
y1 = x ** 2 + x * 2 + 1
y2 = x * 2 + x + 1


plt.figure()

plt.plot(x, y1)

plt.plot(x, y2,color='red',linewidth=2.0,linestyle='--')

plt.xlim((-1, 2))
plt.ylim((-2, 3))

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
# ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]

ax.spines['bottom'].set_position(('data', 0))
# the 1st is in 'outward' | 'axes' | 'data'
# axes: percentage of y axis
# data: depend on y data

ax.yaxis.set_ticks_position('left')
# ACCEPTS: [ 'left' | 'right' | 'both' | 'default' | 'none' ]

ax.spines['left'].set_position(('data',0))
plt.show()
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

plt.yticks([-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
plt.show()
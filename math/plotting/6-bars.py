#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

people = ['Farrah', 'Fred', 'Felicia']
x = np.arange(len(people))

colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
labels = ['apples', 'bananas', 'oranges', 'peaches']

bottom = np.zeros(3)

for i in range(len(fruit)):
    plt.bar(x, fruit[i], bottom=bottom, color=colors[i], label=labels[i], width=0.5)
    bottom += fruit[i]

plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.xticks(x, people)
plt.yticks(np.arange(0, 81, 10))
plt.ylim(0, 80)
plt.legend()

plt.show()

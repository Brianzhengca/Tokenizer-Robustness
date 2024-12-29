import matplotlib.pyplot as plt
categories = ['Normal', 'Perturbed']
values = [0.497, 0.567]
bars = plt.bar(categories, values, color=['skyblue', 'orange'])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, 
             f'{bar.get_height()}', ha='center', va='bottom')
plt.ylabel('Value')
plt.title('Generate Acronym Task')
plt.savefig('results.jpg')
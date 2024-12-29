import matplotlib.pyplot as plt
categories = ['Normal', 'Perturbed']
values = [0.679, 0.789]
bars = plt.bar(categories, values, color=['skyblue', 'orange'])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, 
             f'{bar.get_height()}', ha='center', va='bottom')
plt.ylabel('Value')
plt.title('Count Characters Task')
plt.savefig('results.jpg')
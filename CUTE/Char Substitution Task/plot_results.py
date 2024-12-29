import matplotlib.pyplot as plt
categories = ['Rarely Segmented', 'Commonly Segmented']
values = [0.041, 0.180]
bars = plt.bar(categories, values, color=['skyblue', 'orange'])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, 
             f'{bar.get_height()}', ha='center', va='bottom')
plt.ylabel('Value')
plt.title('Character Substitution Task')
plt.savefig('results.jpg')
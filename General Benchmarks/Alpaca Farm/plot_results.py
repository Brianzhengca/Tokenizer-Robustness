import matplotlib.pyplot as plt
categories = ['Length Controlled', 'Raw']
values = [0.500, 0.438]
bars = plt.bar(categories, values, color=['skyblue', 'orange'])
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, 
             f'{bar.get_height()}', ha='center', va='bottom')
plt.ylabel('Value')
plt.title('Alpaca Farm Winrate Perturbed Against Normal Inputs')
plt.savefig('results.jpg')
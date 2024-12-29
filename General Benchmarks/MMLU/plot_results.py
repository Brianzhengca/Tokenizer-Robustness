import matplotlib.pyplot as plt
import numpy as np

data = {"Math":[0.457, 0.400], "Health":[0.720, 0.597], "Physics":[0.580, 0.494], "Business":[0.826, 0.741], "Biology":[0.808, 0.705], "Chemistry":[0.571, 0.479], "Computer Science":[0.641, 0.512], "Economics":[0.697, 0.539], "Engineering":[0.655, 0.469], "Philosophy":[0.673, 0.499], "Other":[0.728, 0.603], "History":[0.800, 0.705], "Geography":[0.798, 0.682], "Politics":[0.807, 0.701], "Psychology":[0.788, 0.643], "Culture":[0.819, 0.711], "Law":[0.541, 0.469], "STEM":[0.582, 0.492], "Humanities":[0.649, 0.528], "Social Sciences":[0.774, 0.640], "Other (Business, Health, Misc.)":[0.737, 0.618], "Overall":[0.682, 0.566]}

languages = list(data.keys())
values = np.array(list(data.values()))
normal_values = values[:, 0]
perturbed_values = values[:, 1]

x = np.arange(len(languages))
bar_width = 0.4

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - bar_width/2, normal_values, bar_width, label='Normal', color='skyblue')
rects2 = ax.bar(x + bar_width/2, perturbed_values, bar_width, label='Perturbed', color='orange')

ax.set_ylabel('Percentage')
ax.set_title('Normal vs Perturbed Results by Task')
ax.set_xticks(x)
ax.set_xticklabels(languages, rotation=90)
ax.legend()

def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height*100:.1f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.savefig('results.jpg', dpi=300)
plt.show()

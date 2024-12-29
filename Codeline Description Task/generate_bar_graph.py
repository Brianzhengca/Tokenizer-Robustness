import matplotlib.pyplot as plt
import numpy as np

data = {"C++":[0.640, 0.850], "Csharp":[0.654, 0.844], "Java":[0.671, 0.873], "Javascript":[0.630, 0.832], "PHP":[0.659, 0.821], "Python":[0.712, 0.796]}

languages = list(data.keys())
values = np.array(list(data.values()))
normal_values = values[:, 0]
perturbed_values = values[:, 1]

# Number of categories
x = np.arange(len(languages))

# Bar width
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
rects1 = ax.bar(x - bar_width/2, normal_values, bar_width, label='Normal', color='skyblue')
rects2 = ax.bar(x + bar_width/2, perturbed_values, bar_width, label='Perturbed', color='orange')

# Add labels, title, and ticks
ax.set_ylabel('Percentage')
ax.set_title('Normal vs Perturbed Results by Language')
ax.set_xticks(x)
ax.set_xticklabels(languages)
ax.legend()

# Function to add percentage labels on top of each bar
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        # Format as a percentage with one decimal place
        ax.annotate(f'{height*100:.1f}%',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 5), # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add labels to both sets of bars
add_labels(rects1)
add_labels(rects2)

plt.tight_layout()
plt.savefig('results.jpg', dpi=300)
plt.show()
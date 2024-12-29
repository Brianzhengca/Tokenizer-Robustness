import matplotlib.pyplot as plt

# Your data
data = [0.473, 0.449, 0.426, 0.478]
# 0.4584
# 0.4387
# 0.4080
# 0.4810
labels = ["1-2", "2-3", "3-4", "4-5"]

# Create the bar chart
bars = plt.bar(labels, data, color='skyblue', edgecolor='black')

# Add data labels on top of each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
             f"{data[i]:.3f}",  # Format your value as needed
             ha='center', va='bottom')

# Label the axes
plt.xlabel("Length Ratio")
plt.ylabel("Win Rate")

# Optionally, set a title
plt.title("Win Rate vs. Length Ratio")

# You can also set y-axis limits if desired
plt.ylim([0, 1])

plt.savefig("results.jpg", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
#697 Samples
import matplotlib.pyplot as plt

# Your data
data = [0.298, 0.295, 0.288, 0.284]
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
plt.ylabel("Accuracy")

# Optionally, set a title
plt.title("Accuracy vs. Length Ratio")

# You can also set y-axis limits if desired
plt.ylim([0, 1])

plt.savefig("results.jpg", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
#697 Samples
import matplotlib.pyplot as plt

# Define the number of samples and corresponding accuracies
n_samples = [1, 2, 4, 8, 16, 32]
original_acc = [74.51, 74.83, 74.95, 75.40, 77.60, 79.90]
after_finetuning_acc = [75.61, 76.20, 76.38, 78.28, 78.69, 80.16]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot Original CLIP Few-Shot Accuracies
plt.plot(n_samples, original_acc, marker='o', linestyle='-', color='blue', label='Original')

# Plot After Fine-Tuning Learning Rate CIFAR-100 Accuracies
plt.plot(n_samples, after_finetuning_acc, marker='s', linestyle='--', color='green', label='After Fine-Tuning Learning Rate')

# Add title and labels
plt.title('Comparison of Original CLIP Few-Shot and After Fine-Tuning Learning Rate Accuracies')
plt.xlabel('Number of Samples (n)')
plt.ylabel('Accuracy (%)')

# Add grid
plt.grid(True, linestyle='--', alpha=0.5)

# Add legend
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the plot as a PNG file
# plt.savefig('accuracy_comparison_updated.png')

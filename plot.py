import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Data points
x = np.array([99.80,96.28,94.46,91.00,26.34,9.71,84.29,88.01,57.43,38.40,12.71,12.19,12.78,6.78, 13.2, 87.48,91.00,96.09,69.36,37.74,13.30,13.95,9.78,39.24,50.78,46.94,43.8,79.9, 62.0, 31.0, 2.9, 5.7])
y = [0.0,0.3,0.3,1.9,14.5,21.3,1.2,0.6,5.3,7.9,37.9,7.0,3.5,15.1, 13.9, 0.4,0.3,0.3,5.0,21.3,20.3,22.5,27.2,15.0,4.4,5.6,3.4,1.5,3.2, 9.0, 30.1, 20.8]

y = np.array([100 - dd for dd in y])

print(len(x))
print(len(y))

labels = [
    "Claude-2.1", "Claude-3-haiku", "Claude-3-sonnet", "Claude-3-opus", "Gemma-7b", "Gemini-1.0-pro", "Gemini-1.5-flash", "Gemini-1.5-pro*", "GPT-3.5-turbo-0301", "GPT-3.5-turbo-0613", "GPT-3.5-turbo-0125", "GPT-4-0125-preview", "GPT-4-turbo-2024-04-09*", "GPT-4o", "GPT-4o-2024-08-06", "Llama-2-7b", "Llama-2-13b", "Llama-2-70b", "Llama-3-8b", "Llama-3-70b*", "Mistral-small-latest",
    "Mistral-medium-latest", "Mistral-large-latest", "Qwen-1.5-7B", "Qwen-1.5-32B", "Qwen-1.5-72B", "Claude-3.5-sonnet",
    "Gemma-2-9b", "Gemma-2-27b", "Llama-3.1-8B", "Llama-3.1-70B", "Llama-3.1-405B"
]

fig, ax = plt.subplots(figsize=(16, 10))

colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

color_mapping = {
    'Claude': '#D79FDD',
    'Gemini': 'skyblue',
    'GPT-3.5': 'magenta',
    'Llama-3': '#F26374',
    'Llama-2': '#B2D675',
    'Mistral': '#FFED00',
    'Qwen': '#FDB078',
    'GPT-4': 'pink',

}

def get_model_family(label_name):
    if label.startswith("Llama-3"):
      model_family = "Llama-3"
    elif label.startswith("Llama-2"):
      model_family = "Llama-2"
    elif label.startswith("GPT-3.5-turbo"):
      model_family = "GPT-3.5"
    elif label.startswith("GPT-4"):
      model_family = "GPT-4"
    elif label.startswith("Gemini") or label.startswith("Gemma"):
      model_family = "Gemini"
    else:
      model_family = label.split('-')[0]
    return model_family


def get_marker(model):
  # markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', 'x']
  model_family = get_model_family(model)
  marker_mapping = {
    'Claude': 'o',
    'Gemini': 'P',
    'GPT-3.5': '*',
    'GPT-4': 'X',
    'Llama-3': 's',
    'Llama-2': '>',
    'Mistral': '^',
    'Qwen': 'D'
  }

  return marker_mapping[model_family]



already_added_label = set()
for i, (x_val, y_val, label) in enumerate(zip(x, y, labels)):
    model_family = get_model_family(label)
    ax.scatter(x_val, y_val, color=color_mapping[model_family], marker=get_marker(label), s=200, edgecolors='black', linewidths=0.5, label=model_family if model_family not in already_added_label else "")
    already_added_label.add(model_family)

legend_fontsize = 14

label_offset = 2
for label, x_val, y_val in zip(labels, x, y):
    if label == "Llama-3-70b*":
        ax.text(x_val, y_val - 1, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Claude-3-opus":
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val, y_val - 3),
                  textcoords='data', ha='center', va='top', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"))
    elif label == "Claude-3-haiku":
        ax.text(x_val + 1, y_val - 1.5, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Claude-3-sonnet":
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val-2, y_val + 2),
                  textcoords='data', ha='center', va='bottom', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"))
    elif label == "Claude-3.5-sonnet":
        ax.text(x_val - 1.5, y_val, f' {label}', verticalalignment='center', horizontalalignment='right', fontsize=legend_fontsize)
    elif label == "Claude-2.1":
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val-0.5, y_val + 2.5),
                  textcoords='data', ha='center', va='bottom', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"))
        # ax.text(x_val, y_val + 3, f' {label}', verticalalignment='bottom', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Qwen-1.5-32B":
        ax.text(x_val+2, y_val + 1, f' {label}', verticalalignment='bottom', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Qwen-1.5-72B":
        # ax.annotate(label, xy=(x_val, y_val), xytext=(x_val + 3, y_val -2),
        #           textcoords='data', ha='center', va='top', fontsize=12,
        #           arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"))
        ax.text(x_val + 3, y_val -2, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Qwen-1.5-7B":
        ax.text(x_val + 2.5, y_val, f' {label}', verticalalignment='center', horizontalalignment='left', fontsize=legend_fontsize)
    elif label.startswith("GPT-4-0125-preview"):
        ax.text(x_val - 2, y_val - 1, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label.startswith("GPT-4-1106-preview"):
        ax.text(x_val, y_val - 1.5, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label.startswith("GPT-4-turbo-2024-04-09"):
        ax.text(x_val, y_val + 1, f' {label}', verticalalignment='bottom', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == ("GPT-4o"):
        ax.text(x_val, y_val-1, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == ("GPT-4o-2024-08-06"):
        ax.text(x_val, y_val+1, f' {label}', verticalalignment='bottom', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "GPT-3.5-turbo-0125":
        ax.text(x_val, y_val + 1, f' {label}', verticalalignment='bottom', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "GPT-3.5-turbo-0613":
        ax.text(x_val-2.5, y_val+1.5, f' {label}', verticalalignment='center', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "GPT-3.5-turbo-0301":
        ax.text(x_val + 2.5, y_val - 1, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Mistral-large-latest":
        ax.text(x_val + 3 , y_val - 1, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Mistral-medium-latest":
        ax.text(x_val , y_val-1, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Mistral-small-latest":
        ax.text(x_val + 1 , y_val + 1, f' {label}', verticalalignment='center', horizontalalignment='left', fontsize=legend_fontsize)
    elif label == "Gemini-1.5-pro*":
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val-4.5, y_val - label_offset),
                  textcoords='data', ha='center', va='top', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"), zorder=10)
    elif label == "Gemini-1.5-flash":
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val - 4, y_val + 1),
                  textcoords='data', ha='right', va='top', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"))
    elif label == "Gemini-1.0-pro":
        ax.text(x_val+1, y_val, f' {label}', verticalalignment='center', horizontalalignment='left', fontsize=legend_fontsize)
    elif label == "Gemma-7b":
        ax.text(x_val, y_val - 1.2, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Gemma-2-27b":
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val-2, y_val+3),
                  textcoords='data', ha='center', va='top', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"))
    elif label == "Gemma-2-9b":
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val-2, y_val-3),
                  textcoords='data', ha='center', va='top', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"))
    elif label == "Llama-2-70b":
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val+2, y_val - 3),
                  textcoords='data', ha='center', va='top', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"))
        # ax.text(, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Llama-3-8b":
        ax.text(x_val, y_val+1, f' {label}', verticalalignment='bottom', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Llama-3.1-8B":
        ax.text(x_val-0.3, y_val-1, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Llama-3.1-70B":
        ax.text(x_val + 4, y_val-1, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Llama-3.1-405B":
        ax.text(x_val+1.7, y_val+2, f' {label}', verticalalignment='bottom', horizontalalignment='center', fontsize=legend_fontsize)
    elif label == "Llama-2-7b":
        ax.text(x_val-1, y_val + 1, f' {label}', verticalalignment='center', horizontalalignment='right', fontsize=legend_fontsize, zorder=2)
    elif label == "Llama-2-13b":
        # ax.text(x_val - 10, y_val - 3, f' {label}', verticalalignment='bottom', horizontalalignment='center', fontsize=legend_fontsize)
        ax.annotate(label, xy=(x_val, y_val), xytext=(x_val - 11, y_val + 2.5),
                  textcoords='data', ha='center', va='bottom', fontsize=legend_fontsize,
                  arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.0", linestyle="dashed"), zorder=2)
    else:
        ax.text(x_val, y_val - label_offset, f' {label}', verticalalignment='top', horizontalalignment='center', fontsize=legend_fontsize)


# Linear regression to compute the slope
x_reshaped = x.reshape(-1, 1)
reg = LinearRegression().fit(x_reshaped, y)
slope = reg.coef_[0]

# Coordinates of GPT-3.5-turbo-0613
gpt_3_5_turbo_0613_x = x[labels.index("GPT-3.5-turbo-0613")]
gpt_3_5_turbo_0613_y = y[labels.index("GPT-3.5-turbo-0613")] + 2

# Calculate the new intercept to move the line to cross GPT-3.5-turbo-0613
intercept = gpt_3_5_turbo_0613_y - slope * gpt_3_5_turbo_0613_x

# Create line using the computed slope and new intercept
x_line = np.linspace(0, 100, 500)
y_line = slope * x_line + intercept




# Plot the adjusted line
# ax.plot(x_line, y_line, color='black', linestyle='--', linewidth=1)

# Fill the area above the line
# ax.fill_between(x_line, y_line, y2=105, where=(y_line < 105), color='skyblue', alpha=0.2)

coefficients = np.polyfit(x, y, 2)  # '2' for quadratic
polynomial = np.poly1d(coefficients)

# Generate a sequence of x values for plotting the fitted curve
x_line = np.linspace(min(x), max(x), 300)
y_line = polynomial(x_line)

plt.plot(x_line, y_line, 'r:', linestyle=(0, (5, 8)), color='blue', alpha=0.5)

arrow_color='#f66054'

# Add the arrows
plt.annotate(
    '', xy=(1, 85), xytext=(1, 65),
    arrowprops=dict(facecolor=arrow_color, edgecolor=arrow_color, shrink=0.05, width=2, headwidth=8, alpha=0.8)
)
plt.text(2, 75, 'Safe', color=arrow_color, rotation=90, verticalalignment='center', fontsize=legend_fontsize)

plt.annotate(
    '', xy=(65, 63.5), xytext=(30, 63.5),
    arrowprops=dict(facecolor=arrow_color, edgecolor=arrow_color, shrink=0.05, width=2, headwidth=8, alpha=0.8)
)
plt.text(42, 64.5, 'Over-Refusal', color=arrow_color, verticalalignment='center', fontsize=legend_fontsize)

# Manually setting the y-axis limit
ax.set_ylim(bottom=61, top=104)  # Setting the top y-value to 105
ax.set_xlim(0, 104)  # Setting the top y-value to 105

# Add a legend in the lower right corner of the plot
ax.legend(loc='lower right', fontsize=legend_fontsize)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlabel('Over-Refusal Prompts Rejection Rate', fontsize=18)
ax.set_ylabel('Toxic Prompts Rejection Rate', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=14)

plt.savefig('overall_x_y_plot.pdf', format='pdf', bbox_inches='tight', dpi=1000)
plt.savefig('overall_x_y_plot.png', format='png', bbox_inches='tight', dpi=600)
plt.show()

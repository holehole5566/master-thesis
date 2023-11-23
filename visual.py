import numpy as np
import json
import matplotlib.pyplot as plt
def main():
    quads = ["Q1_gpt","Q1_feature","Q1_raw"]
    quad_settings = {
        "Q1_gpt" : ("red", "Q1"),
        "Q1_feature" : ("blue", "Q1_gen"),
        "Q1_raw" : ("green", "Q1_raw"),
        "Q1_11" : ("yellow", "Q4")
    }
    # Creating the plot and defining the four quadrants
    fig, ax = plt.subplots()
    for quadrant in quads:

        with open("result_{}.json".format(quadrant), "r") as f:
            data = json.load(f)

        # Generating sample data for each quadrant
        x_values = data["x"]
        y_values = data["y"]

        
        color = quad_settings[quadrant][0]
        label = quad_settings[quadrant][1]
        # Scatter plot points for each quadrant
        ax.scatter(x_values, y_values, color=color, label=label)

        # Setting labels and title
        ax.set_xlabel('arousal')
        ax.set_ylabel('valence')
        ax.set_title('thayer mood model')

        # Adding lines to divide the quadrants
        ax.axhline(0, color='black',linewidth=0.5)
        ax.axvline(0, color='black',linewidth=0.5)

        # Set the plot limits to center at (0, 0)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        # Displaying the plot
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
import json
import matplotlib.pyplot as plt

def check_quadrant(x_values, y_values):
    q1, q2, q3, q4 = 0, 0, 0, 0
    for x, y in zip(x_values, y_values):
        if x > 0 and y > 0:
            q1 += 1
        elif x < 0 and y > 0:
            q2 += 1
        elif x < 0 and y < 0:
            q3 += 1
        elif x > 0 and y < 0:
            q4 += 1
    return q1, q2, q3, q4
        
def main():
    q = "Q3"
    quads = ["raw","gpt","feature"]
    quad_settings = {
        "raw" : ("orange", "raw"),
        "gpt" : ("blue", "gpt"),
        "feature" : ("green", "feature"),
        "gt" : ("red", "gt")
    }
    # Creating the plot and defining the four quadrants
    fig, ax = plt.subplots()
    # Setting labels and title
    ax.set_xlabel('valence')
    ax.set_ylabel('arousal')

    # Adding lines to divide the quadrants
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    title = ""
    # Set the plot limits to center at (0, 0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    for quadrant in quads:

        with open("result_{}_{}.json".format(q, quadrant), "r") as f:
            data = json.load(f)

        x_values = data["x"]
        y_values = data["y"]

        q1, q2, q3, q4 = check_quadrant(x_values, y_values)
        quad_count = {
            "Q1" : q1,
            "Q2" : q2,
            "Q3" : q3,
            "Q4" : q4
        }   
        title += quadrant + ": " + str(round(100 * quad_count[q] / len(x_values), 2)) + "%  "
        
        color = quad_settings[quadrant][0]
        label = quad_settings[quadrant][1]
        # Scatter plot points for each quadrant
        ax.scatter(x_values, y_values, color=color, label=label)

       
        # Displaying the plot
    ax.set_title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

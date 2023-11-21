import json

def main():
    quadrant = "Q1"
    x = []
    y = []
    with open("result_generated.json", "r") as f:
        data = json.load(f)
    values = data["results"]
    for value in values:
        h_a = value["high arousal"]
        l_a = value["low arousal"]
        p_v = value["positive valence"]
        n_v = value["negative valence"]
        x.append(int(h_a - l_a)/100)
        y.append(int(p_v - n_v)/100)
    coordinate = {
        "x" : x,
        "y" : y
    }
    with open("result_{}.json".format(quadrant), 'w') as f:
        json.dump(coordinate, f)

if __name__ == "__main__":
    main()

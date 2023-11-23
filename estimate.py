import json

def main():
    quadrant = "Q1"
    valences = []
    arousals = []
    with open("result_generated.json", "r") as f:
        data = json.load(f)
    values = data["results"]
    for value in values:
        happy = value["happy mood"] # arousal:1 valence:1
        sad = value["sad mood"] # arousal:-1 valence:-1
        angry = value["angry mood "] # arousal:1 valence:-1
        relaxed = value["relaxed mood"] # arousal:-1 valence:1
        val = happy - sad - angry + relaxed
        arou = happy - sad + angry - relaxed
        valences.append(round(val, 2))
        arousals.append(round(arou, 2))
    coordinate = {
        "x" : valences,
        "y" : arousals
    }
    with open("result_{}.json".format(quadrant), 'w') as f:
        json.dump(coordinate, f)

if __name__ == "__main__":
    main()

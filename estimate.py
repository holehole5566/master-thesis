import json

def main():
    quadrant = "Q4"
    valences = []
    arousals = []
    with open("result_generated.json", "r") as f:
        data = json.load(f)
    values = data["results"]
    for value in values:
        happy = value["happy song"] # arousal:1 valence:1
        excited = value["excited song"] # arousal:1 valence:1
        pleased = value["pleased song"] # arousal:1 valence:1
        bored = value["bored song"] # arousal:-1 valence:-1
        sleepy = value["sleepy song"] # arousal:-1 valence:-1
        sad = value["sad song"] # arousal:-1 valence:-1
        angry = value["angry song"] # arousal:1 valence:-1
        nervous = value["nervous song"] # arousal:1 valence:-1
        annoying = value["annoying song"] # arousal:1 valence:-1
        peaceful = value["peaceful song"] # arousal:-1 valence:1
        calm = value["calm song"] # arousal:-1 valence:1
        relaxed = value["relaxed song"] # arousal:-1 valence:1

        val = happy + excited + pleased - sad - bored - sleepy - angry - nervous - annoying + relaxed + peaceful + calm
        arou = happy + excited + pleased - sad -bored - sleepy + angry + nervous + annoying - relaxed - peaceful - calm
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

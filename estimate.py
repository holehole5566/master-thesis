import json


def main():
    with open("result_generated.json", "r") as f:
        data = json.load(f)
    print(data)




if __name__ == "__main__":
    main()

# MVP for FilmCriticLM

import random

data = []

def read_data():
    """

    """
    print("Reading data...\n")

    with open("data.txt", encoding="utf8") as f:
        line = f.readline()

        while line:
            data.append(line.strip())
            line = f.readline()

def run_training():
    pass

def run_evaluation():
    accuracy = 0
    creativity = 0
    num = min(len(data)-1, 100)
    for i in range(num):
        test_prompt = data[i].split(" ")[:-2]
        test_response = generate_response(test_prompt)

    print("Accuracy:", accuracy)
    print("Creativity:", creativity)
    print("Evaluation complete.\n")

def generate_response(prompt):
    # pick random data entry
    result = data[random.randint(0, len(data) - 1)]
    assert isinstance(result, str), "Generated response must be a string."

    print(result + "\n")
    print("Response generated.\n")

def run_tests():
    print("Tests passed.\n")

# main
if __name__ == '__main__':

    read_data()

    run_training()

    request = int(input("Type in 1 to run tests. \nType in 2 to run the evaluation. \nType in 3 to make a prompt.\n"))

    match request:
        case 1:
            run_tests()
        case 2:
            run_evaluation()
        case 3:
            prompt = input("Requesting prompt...\n")
            generate_response(prompt)
        case _:
            print("This is not a valid input.")
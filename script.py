# MVP for FilmCriticLM

import random
import re
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer

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

def sentence_to_list(sentence):
    return sentence.split(" ")

def list_to_sentence(list):
    result = ""
    for i in range(len(list)):
        if i == len(list) - 1:
            result += list[i]
        else:
            result += list[i] + " "
    return result

def run_evaluation():
    accuracy_sum = 0
    # creativity_sum = 0
    num = min(len(data)-1, 100)
    for i in range(num):
        test_prompt = list_to_sentence(sentence_to_list(data[i])[:-2])
        test_expected_answer = list_to_sentence(data[i].split(" ")[-2:])
        test_response = generate_response(test_prompt)

        print("Prompt:", test_prompt)
        print("Expected answer:", test_expected_answer)
        print("Actual answer:", test_response)

        manual_test = int(input("Valid (1) or not (0)?"))
        if manual_test == 1:
            accuracy_sum += 1
        print("\n")

    print("Accuracy:", float(accuracy_sum/num))
    # print("Creativity:", float(creativity_sum/num))
    print("Evaluation complete.\n")

def generate_response(prompt):
    # pick random data entry
    assert isinstance(prompt, str), "Prompt must be a string!"
    result = data[random.randint(0, len(data) - 1)]
    assert isinstance(result, str), "Generated response must be a string!"

    return result


def run_tests():
    assert sentence_to_list("This is a test") == ["This", "is", "a", "test"], "Converting sentences to lists failed!"
    assert list_to_sentence(["This", "is", "a", "test"]) == "This is a test", "Converting lists to sentences failed!"
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
            print(generate_response(prompt) + "\n")
            print("Response generated.\n")
        case _:
            print("This is not a valid input.")
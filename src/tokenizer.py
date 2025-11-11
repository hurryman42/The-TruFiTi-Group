
def return_base_vocabulary():
    result = {}

    for i in range(256):
        result[i] = i.to_bytes()

    return result.copy()

def word_to_bytes(word):
    return bytes(word, "utf-8")

def is_prefix_of(list1, list2):
    if len(list1) <= len(list2):
        return list2[:len(list1)] == list1
    else:
        return False

def byte_pair_encode(sentence):
    vocab = return_base_vocabulary()
    counter = {}

    words = sentence.strip().split(" ")
    for word in words:
        for i in range(len(word)-1):
            bytepair = bytes(word[i], "utf-8") + bytes(word[i+1], "utf-8")
            if bytepair in counter:
                counter[bytepair] += 1
            else:
                counter[bytepair] = 1

    new_bytepair = max(counter, key=counter.get)
    vocab[len(vocab)] = max(counter, key=counter.get)

    new_sentence = ""
    i = 0
    while i < len(sentence)-1:
        bytepair = bytes(sentence[i], "utf-8") + bytes(sentence[i+1], "utf-8")
        if bytepair == new_bytepair:
            new_sentence += "<256>"
            i += 2
        else:
            new_sentence += sentence[i]
            i += 1
    new_sentence += sentence[len(sentence)-1]

    return new_sentence

if __name__ == '__main__':
    print(byte_pair_encode("the cat in the hat"))
    print(is_prefix_of(bytes("aab", "utf-8"), bytes("aabbba", "utf-8")))
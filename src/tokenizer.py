import sys

def return_base_vocabulary():
    """
    Returns a dictionary that is the basis for a vocabularies used to encode Strings.
    """
    result = {}

    for i in range(256):
        result[i] = i.to_bytes()

    return result.copy()

def is_prefix_of(list1, list2):
    """
    Returns True if list1 is a prefix of list2.
    """
    if len(list1) <= len(list2):
        return list2[:len(list1)] == list1
    else:
        return False

def encode_with_vocabulary(input, vocab):
    """
    Encodes a given input with the given vocabulary and returns it as a vector.
    """
    result = []

    i = 0
    while i < len(input):
        temp = -1
        temp_len = 0
        for key, value in vocab.items():
            if is_prefix_of(value, bytes(input[i:], "utf-8")) and (key > temp):
                temp = key
                temp_len = len(value)
        result.append(temp)
        i += temp_len

    return result.copy()

def decode_with_vocabulary(input, vocab):
    """
    Decodes a given input with the given vocabulary and returns it as a string.
    """
    result = bytes("", "utf-8")

    for element in input:
        result += vocab[element]

    return result.decode("utf-8")

def byte_pair_encode(sentences, vocab_size=256):
    """
    Returns a vocabulary that is the result of the byte-level byte pair encoding algorithm.
    """
    assert vocab_size >= 256, "Vocabulary size must be at least 256."

    vocab = return_base_vocabulary()

    for i in range(vocab_size - 256):
        counter = {}

        for sentence in sentences:
            words = sentence.strip().split(" ")

            for word in words:
                encoded_word = encode_with_vocabulary(word, vocab)
                for i in range(len(encoded_word)-1):
                    pair = str(encoded_word[i]) + "-" + str(encoded_word[i+1]) #necessary because lists are not hashable in Python
                    if pair in counter:
                        counter[pair] += 1
                    else:
                        counter[pair] = 1

        dominant = max(counter, key=counter.get)
        dominant_pair = dominant.split("-")

        byte_pair = vocab[int(dominant_pair[0])] + vocab[int(dominant_pair[1])]

        vocab[len(vocab)] = byte_pair

    return vocab.copy()

if __name__ == '__main__':

    example_sentence = "the cat hat"
    print(encode_with_vocabulary(example_sentence, return_base_vocabulary()))

    print(b'h'.hex())

    print("Input Sentence:", example_sentence)
    test_vocab = byte_pair_encode([example_sentence], 257)
    print("Byte-Level BPE vocabulary:", test_vocab)
    test_encode = encode_with_vocabulary(example_sentence, test_vocab)
    print("Encoded vector: ", test_encode)
    print("Decoded string: ", decode_with_vocabulary(test_encode, test_vocab))

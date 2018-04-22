import os
import math
import numpy
import codecs


def hash_string(s):
    return abs(hash(s)) % (10 ** 8)


def xavier_vector(word, D=300):
    """
    Returns a D-dimensional vector for the word.
    We hash the word to always get the same vector for the given word.
    """
    seed_value = hash_string(word)
    numpy.random.seed(seed_value)

    neg_value = - math.sqrt(6)/math.sqrt(D)
    pos_value = math.sqrt(6)/math.sqrt(D)

    rsample = numpy.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = numpy.linalg.norm(rsample)
    rsample_normed = rsample/norm

    return rsample_normed


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word] ** 2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_word_vectors(file_destination, primary_language="english"):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector dimensionality.
    """
    print "Loading pretrained word vectors from", file_destination,\
        "- treating", primary_language, "as the primary language."

    word_dictionary = {}
    lp = {"english": u"en_", "german": u"de_", "italian": u"it_",
          "russian": u"ru_", "sh": u"sh_", "bulgarian": u"bg_",
          "polish": u"pl_", "spanish": u"es_", "french": u"fr_",
          "portuguese": u"pt_", "swedish": u"sv_", "dutch": u"nl_"}
    language_key = lp[primary_language]

    f = codecs.open(file_destination, 'r', 'utf-8')
    for line in f:
        line = line.split(" ", 1)
        transformed_key = line[0].lower()
        if language_key in transformed_key:
            transformed_key = transformed_key.replace(language_key, "")
            try:
                transformed_key = unicode(transformed_key)
            except:
                print "Can't convert the key to unicode:", transformed_key
            word_dictionary[transformed_key] = numpy.fromstring(line[1], dtype="float32", sep=" ")
            if word_dictionary[transformed_key].shape[0] != 300:
                print transformed_key, word_dictionary[transformed_key].shape
    print len(word_dictionary), "vectors loaded from", file_destination

    return normalise_word_vectors(word_dictionary)


def download_word_vectors(file_path):
    print "Vectors not there, downloading small Paragram and putting it there."
    os.system("wget https://mi.eng.cam.ac.uk/~nm480/prefix_paragram.txt")
    os.system("mv prefix_paragram.txt " + file_path)

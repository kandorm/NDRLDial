import codecs
import json


class DataReader(object):
    def __init__(self, ontology_file, corpus_file):

        self.ontology = json.load(codecs.open(ontology_file, "r", "utf-8"))
        self.corpus = json.load(codecs.open(corpus_file, 'r', 'utf-8'))

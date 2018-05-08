import codecs
import json


class DataReader(object):
    def __init__(self, ontology_file, corpus_file):

        self.ontology = json.load(codecs.open(ontology_file, "r", "utf-8"))
        self.corpus = self._load_corpus(corpus_file)

    def _load_corpus(self, corpus_file):
        data = json.load(codecs.open(corpus_file, 'r', 'utf-8'))
        corpus = []

        #data = data[-len(data)/5:]

        for dialog in data:
            user_transcript = []
            goal = {}
            finished = dialog['finished']

            dial = dialog['dial']
            for turn in dial:
                user_transcript.append(turn['usr']['transcript'])

            dial_constraints = dialog['goal']['constraints']
            for const in dial_constraints:
                goal[const[0]] = const[1]
            goal['request'] = dialog['goal']['request-slots']

            corpus.append((user_transcript, goal, finished))

        return corpus









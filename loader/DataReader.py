import codecs
import json


class DataReader(object):
    def __init__(self, ontology_file, corpus_file):

        self.ontology = json.load(codecs.open(ontology_file, "r", "utf-8"))
        self.corpus = self._load_corpus(corpus_file)

    def _load_corpus(self, corpus_file):
        data = json.load(codecs.open(corpus_file, 'r', 'utf-8'))
        corpus = []

        data = data[:len(data)/5]

        for dialog in data:
            user_transcript = []
            goal = {'request': []}
            finished = dialog['finished']

            dial = dialog['dial']
            for turn in dial:
                user_transcript.append(turn['usr']['transcript'])
                slu = turn['usr']['slu']
                for s_s in slu:
                    if s_s['act'] == 'inform':
                        for s_v in s_s['slots']:
                            if s_v[0] != 'slot':
                                goal[s_v[0]] = s_v[1]
                            else:
                                goal['request'].append(s_v[1])
                    elif s_s['act'] == 'request':
                        for s_v in s_s['slots']:
                            goal['request'].append(s_v[1])

            goal['request'] = list(set(goal['request']))

            corpus.append((user_transcript, goal, finished))

        return corpus









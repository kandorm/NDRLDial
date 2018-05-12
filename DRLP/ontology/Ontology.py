from DRLP.ontology.FlatOntologyManager import FlatOntologyManager

global_ontology = None


def init_global_ontology():
    global global_ontology
    global_ontology = FlatOntologyManager()

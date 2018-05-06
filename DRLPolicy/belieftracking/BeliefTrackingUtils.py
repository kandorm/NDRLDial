import json


def addprob(sluhyps, hyp, prob):
    """
    Add prob to hyp in slu hypotheses

    :param sluhyps: slu hypotheses
    :type sluhyps: dict

    :param hyp: target hypothesis
    :type hyp: string

    :param prob: probability to be added
    :type prob: float

    :return: dict -- slu hypotheses
    """
    score = min(1.0, float(prob))
    sluhyps[json.dumps(hyp)] += score
    return sluhyps


def normaliseandsort(slu_hyps):
    """
    Normalise and sort the given slu hypotheses

    :param slu_hyps: slu hypotheses
    :type slu_hyps: dict

    :return: list -- list of normalised hypotheses
    """
    result = []
    sorted_hyps = slu_hyps.items()
    sorted_hyps.sort(key=lambda x: -x[1])
    total_score = sum(slu_hyps.values())
    for hyp, score in sorted_hyps:
        if total_score == 0:
            result.append({"score": 0, "slu-hyp": json.loads(hyp)})
        else:
            result.append({"score": min(1.0, score/total_score), "slu-hyp": json.loads(hyp)})
    return result


def transform_act(act, value_trans, ontology=None, user=True):
    """
    Normalise and sort the given slu hypotheses

    :return: dict -- transformed dialogue act
    """
    if user:
        act_without_null = []
        for this_act in act:
            # another special case, to get around deny(name=none,name=blah):
            if this_act["act"] == "deny" and this_act["slots"][0][1] == "none":
                continue
            # another special case, to get around deny(name=blah,name):
            if this_act["act"] == "inform" and this_act["slots"][0][1] is None:
                continue
            # another special case, to get around deny(name,name=blah):
            if this_act["act"] == "deny" and this_act["slots"][0][1] is None:
                continue
            act_without_null.append(this_act)
        act = act_without_null

    # one special case, to get around confirm(type=restaurant) in Mar13 data:
    if not user and ontology is not None and "type" not in ontology["informable"]:
        for this_act in act:
            if this_act["act"] == "expl-conf" and this_act["slots"] == [["type", "restaurant"]]:
                act = [{"act": "confirm-domain", "slots": []}]

    for i in range(len(act)):
        for j in range(len(act[i]["slots"])):
            act[i]["slots"][j][:] = [value_trans[x] if x in value_trans.keys() else x for x in act[i]["slots"][j]]

    # remove e.g. phone=dontcare and task=find
    if ontology is not None:
        new_act = []
        for a in act:
            new_slots = []
            for slot, value in a["slots"]:
                keep = True
                if slot not in ["slot", "this"] and (slot not in ontology["informable"]):
                    if user or (slot not in ontology["requestable"]+["count"]):
                        keep = False
                if keep:
                    new_slots.append((slot, value))
            if len(a["slots"]) == 0 or len(new_slots) > 0:
                a["slots"] = new_slots
                new_act.append(a)
    else:
        new_act = act

    return new_act

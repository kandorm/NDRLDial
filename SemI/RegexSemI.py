import re


class SemI(object):

    def decode(self, asr_obs, sys_act):
        pass


class RegexSemI(SemI):

    def __init__(self):
        self.rTYPE = 'restaurant'
        self._init_re()

    def _init_re(self):
        self.r_reqalts = "(\b|^|\ )((something|anything)\ else)|(different(\ one)*)|(another\ one)|(alternatives*)"
        self.r_reqalts += "|(other options*)|((don\'*t|do not) (want|like)\ (that|this)(\ one)*)"
        self.r_reqalts += "|(others|other\ " + self.rTYPE + "(s)?)"

    def decode(self, obs, sys_act=None):
        self.usr_intent = {}

        if isinstance(obs, str):
            obs = [obs]
        elif not isinstance(obs, list):
            print 'RegexSemI->decode() obs is not str and list'
            return self.usr_intent

        for ob in obs:
            if isinstance(ob, tuple):
                sentence, sentence_prob = ob[0], ob[1]
            elif isinstance(ob, str):
                sentence, sentence_prob = ob, None
            else:
                print 'RegexSemI->decode() item of obs is not tuple and str'
                return self.usr_intent
            assert (isinstance(sentence, str) or isinstance(sentence, unicode))

            self.decode_single_hypothesis(sentence, sentence_prob, sys_act)

        return self.usr_intent

    def decode_single_hypothesis(self, utt, utt_prob=None, sys_act=None):
        self._decode_reqalts(utt, utt_prob)

    def _decode_reqalts(self, utt, utt_prob=None):
        if self._check(re.search(self.r_reqalts, utt, re.I)):
            if 'reqalts' in self.usr_intent:
                if utt_prob is None:
                    self.usr_intent['reqalts'] += 1.0
                else:
                    self.usr_intent['reqalts'] += utt_prob
            else:
                if utt_prob is None:
                    self.usr_intent['reqalts'] = 1.0
                else:
                    self.usr_intent['reqalts'] = utt_prob

    def _check(self, re_object):
        if re_object is None:
            return False
        for o in re_object.groups():
            if o is not None:
                return True
        return False

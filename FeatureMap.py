# -*- coding: utf-8 -*-
"""
Data structure for collecting input and output for ML models from XIGT IGT
data. Also collects counts of input and output instances.

@author: aymm
"""
import re
import sys
import string
import corenlp
import time

class FeatureMap:
    def __init__(self, glosser=None):
        self._parent = glosser
        self.igt_ids = []
        self.set_labels = []
        self.set_features = []
        self.labels2feats = dict() # ex: {"gls":{"source1":[(igtid, index2),...], "source2":[(igtid, index2)]}}
        self.feats2labels = dict() # ex: {"src":{"gloss1":[(igtid, index2),...], "gloss2":[(igtid, index2)]}}
        self.norm_map = dict()
        self.curr_map = None

    @property
    def labels():
        return self.set_labels

    @property
    def features():
        return self.set_features

    def get_feats(self, indices):
        return [self.set_features[index] for index in indices]

    def get_labels(self, indices):
        return [self.set_labels[index] for index in indices]

    def _rm_mapping(self, map, key, entry, igt_id):
        if key in map:
            entry_map = map.get(key)
            if entry in entry_map:
                loc_list = entry_map[entry]
                rm_list = []
                for loc_pair in loc_list:
                    if loc_pair[0] == igt_id:
                        rm_list.append(loc_pair)
                for rm_pair in rm_list:
                    sys.stderr.write("Removed entry: " + str(loc_list.remove(rm_pair)) + "\n")
                if len(loc_list) == 0:
                    entry_map.pop(entry)
            if len(map[key]) == 0:
                map.pop(key)

    def contains_match(self, src, gls, indices):
        if gls in self.labels2feats:
            feats = self.labels2feats[gls]
            for source in feats:
                if source == src:
                    for instance in feats[source]:
                        index = self.igt_ids.index(instance[0])
                        if index in indices:
                            return True
        return False

    def contains_output(self, gls, indices):
        if gls in self.labels2feats:
            for source in self.labels2feats[gls]:
                for instance in self.labels2feats[gls][source]:
                    index = self.igt_ids.index(instance[0])
                    if index in indices:
                        return True
        return False

    def contains_input(self, src, indices):
        if src in self.feats2labels:
            for gloss in self.feats2labels[src]:
                for instance in self.feats2labels[src][gloss]:
                    index = self.igt_ids.index(instance[0])
                    if index in indices:
                        return True
        return False

    def normal_form(self, gls):
        if gls in self.norm_map:
            return self.norm_map[gls]
        else:
            return None

    def instances(self, gls):
        if gls in self.labels2feats:
            inst = 0
            for src in self.labels2feats[gls]:
                inst += len(self.labels2feats[gls][src])
            return inst
        else:
            return -1

    def most_likely(self, src):
        if src in self.feats2labels:
            most_gls = ""
            support = -1
            for gls in self.feats2labels[src]:
                curr_list = self.feats2labels[src][gls]
                if len(curr_list) > support:
                    support = len(curr_list)
                    most_gls = gls
            return most_gls
        else:
            return None

    def get_gls_set(self, src, indices):
        gls_set = set()
        if src in self.feats2labels:
            for gloss in self.feats2labels[src]:
                for instance in self.feats2labels[src][gloss]:
                    index = self.igt_ids.index(instance[0])
                    if index in indices:
                        gls_set.add(gloss)
        return gls_set

    def get_src_set(self, gls, indices):
        src_set = set()
        if gls in self.labels2feats:
            for source in self.labels2feats[gls]:
                for instance in self.labels2feats[gls][source]:
                    index = self.igt_ids.index(instance[0])
                    if index in indices:
                        src_set.add(source)
        return src_set

    def add_mapping(self, map, key, entry, indices):
        if key in map:
            entry_map = map.get(key)
            if entry in entry_map:
                entry_map[entry].append(indices)
            else:
                entry_map[entry] = [indices]
        else:
            map[key] = {entry: [indices]}

    def normalize(self):
        for key in self.labels2feats:
            first_idx = key.find(".")
            if first_idx != -1 and first_idx != len(key) - 1:
                removed = key.replace(".", "")
                if removed in self.labels2feats:
                    if removed in self.norm_map:
                        self.norm_map[removed].append(key)
                    else:
                        self.norm_map[removed] = [key]

        for short in self.norm_map:
            long_list = self.norm_map[short]
            if len(long_list) == 1:
                long = long_list[0]
                self._move_instances(short, long)
            else:
                long = ""
                gram_count = 1
                for gls in long_list:
                    grams = gls.count('.')
                    if grams > gram_count:
                        long = gls
                        gram_count = grams
                for gls in long_list:
                    if gls != long:
                        self._move_instances(gls, short)
                self._move_instances(short, long)
        return self.norm_map

    def _move_instances(self, from_gls, to_gls):
        for src in self.labels2feats[from_gls]:
            for indices in self.labels2feats[from_gls][src]:
                sys.stderr.write(str(self.labels2feats[from_gls][src]))
                igt_idx = self.igt_ids.index(indices[0])
                gls_idx = indices[1]
                labels = self.set_labels[igt_idx]
                labels[gls_idx] = to_gls
                self.add_mapping(self.feats2labels, src, to_gls, indices)
                self.add_mapping(self.labels2feats, to_gls, src, indices)
            self.feats2labels[src].pop(from_gls)
        self.labels2feats.pop(from_gls)

    def dump(self):
        data = dict()
        data["labels"] = self.set_labels
        data["features"] = self.set_features
        data["labels2feats"] = self.labels2feats
        data["feats2labels"] = self.feats2labels
        return data

    def load(self, data):
        self.set_labels = data["labels"]
        self.set_features = data["features"]
        self.labels2feats = data["labels2feats"]
        self.feats2labels = data["feats2labels"]

    def get_value(self, igt, tier, id):
        return self._parent.get_value(igt, tier, id)

    def __str__(self):
        output = "\tMap Contents:\n"
        for i in range(len(self.set_labels)):
            label = self.set_labels[i]
            output += "\t\IGTid: {0:>10}\n".format(self.igt_ids[i])
            output += "\t\toutput: {0:>10}\n".format(str(label))
            for feat in self.set_features[i]:
                output += "\t\t\tinput: {0}\n".format(str(feat))
        output += "\n\tGloss to Source Counts:\n"
        for label, counts in sorted(self.labels2feats.items()):
            output += "\t\tgloss: {0:>10} counts: {1}\n".format(str(label), str(counts))
        output += "\n\tSource to Gloss Counts:\n"
        for label, counts in sorted(self.feats2labels.items()):
            output += "\t\tsource: {0:>10} counts: {1}\n".format(str(label), str(counts))
        return output


"""
A subclass for holding the mapping from the source line to the gloss line.
Uses the aligned gloss as the label. Uses the morpheme string, the source word
string, whether the previous and following morphemes are attached or not, and
the previous and following source words as features.

Note that morphemes that align to multiple glosses receive all of the glosses
as a single label connected by '.'. These labels are normalized alphabetically
such that '3sg.past' and 'past.3sg' are both treated as 'past.3sg'.
"""
class SrcGls_Map(FeatureMap):
    def __init__(self, glosser=None):
        super().__init__(glosser)

    def rm_igt(self, igt):
        index = -1
        if igt.id in self.igt_ids:
            index = self.igt_ids.index(igt.id)
            self.igt_ids.pop(index)
            labels = self.set_labels.pop(index)
            features = self.set_features.pop(index)
            for i in range(len(labels)):
                source = features[i]['m_i']
                sys.stderr.write("Removing mapping:" + labels[i] + " - " + source + "\n")
                super()._rm_mapping(self.labels2feats, labels[i], source, igt.id)
                sys.stderr.write("Removing mapping:" + source + " - " + labels[i] + "\n")
                super()._rm_mapping(self.feats2labels, source, labels[i], igt.id)
        return index

    def add_igt(self, igt):
        igt_labels, igt_features = [], []
        m_len = len(igt.get('m').items)
        w_len = len(igt.get('w').items)
        gls_src_mapping, src_gls_mapping = [], []

        for index in range(m_len):
            item = igt.get('m')[index]
            src = self.get_value(igt, 'm', item.id)
            referrers = igt.referrers(item.id)
            gls_ids = []
            if 'alignment' in referrers:
                gls_ids = referrers['alignment']
                gls_ids = sorted(gls_ids)
            if gls_ids:
                gls = ''
                gls_ids_len = len(gls_ids)
                for gls_id_idx in range(gls_ids_len):
                    if igt.get('g').get(gls_ids[gls_id_idx]):
                        gls += self.get_value(igt, 'g', gls_ids[gls_id_idx])
                        if index != gls_ids_len - 1 and not gls.endswith('.'):
                            gls += '.'
                if gls.endswith('.'):
                    gls = gls[:len(gls)-1]
            elif src in string.punctuation:
                gls = '==PUNC=='
            else:
                gls = '==NONE=='

            self.curr_map = {}

            # current morpheme
            self.curr_map['m_i'] = src
            w_id = igt.referents(item.id)['segmentation'][0]

            # current word
            word = src
            word_idx = -1
            if w_id:
                word = self.get_value(igt, 'w', w_id)
                word_idx = igt.get('w').index(igt.get('w').get(w_id))
            self.curr_map['w_i'] = word

            # if following morpheme is attached
            self.define_m_feat(index == m_len - 1, 'i+1', index + 1, w_id, igt)

            # if previous morpheme is attached
            self.define_m_feat(index == 0, 'i-1', index - 1, w_id, igt)

            # word prior
            self.define_w_feat(word_idx == 0, 'i-1', 'BOS', word_idx - 1, igt)

            # word following
            self.define_w_feat(word_idx == w_len - 1, 'i+1', 'EOS', word_idx + 1, igt)

            igt_features.append(self.curr_map)
            gls_idx = len(igt_labels)
            igt_labels.append(gls)
            gls_src_mapping.append((gls, src, (igt.id, gls_idx)))
            src_gls_mapping.append((src, gls, (igt.id, gls_idx)))
            self.curr_map = None
        self.igt_ids.append(igt.id)
        for mapping in gls_src_mapping:
            self.add_mapping(self.labels2feats, mapping[0], mapping[1], mapping[2])
        for mapping in src_gls_mapping:
            self.add_mapping(self.feats2labels, mapping[0], mapping[1], mapping[2])
        self.set_labels.append(igt_labels)
        self.set_features.append(igt_features)

    def define_m_feat(self, condition, cond_str, index, w_id, igt):
        if condition:
            self.curr_map['m_' + cond_str] = '==NONE=='
        else:
            new_id = igt.get('m')[index].id
            if igt.referents(new_id)['segmentation'][0] == w_id:
                self.curr_map['m_' + cond_str] = self.get_value(igt, 'm', new_id)
            else:
                self.curr_map['m_' + cond_str] = '==NONE=='

    def define_w_feat(self, condition, cond_str, cond_val, index, igt):
        if condition:
            self.curr_map['w_' + cond_str] = cond_val
        else:
            new_id = igt.get('w')[index].id
            self.curr_map['w_' + cond_str] = self.get_value(igt, 'w', new_id)



"""
A subclass for holding the mapping from the translation line to the gloss line.
Uses the gloss as the label as defined by the bilingual alignment (id: a) line from each
translation line, or '==NA==' for 'no alignment'. Uses the translation word string, the
translation word part of speech, and the translation word dependency structure as features,
if they are present.
"""
class TwGls_Map(FeatureMap):
    def __init__(self, glosser=None):
        super().__init__(glosser)
        self.lemmatizer = corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma".split(), timeout=30000, memory='8G')

    def get_translation(self, igt):
        return self._parent.get_translation(igt)

    def get_lemmalist(self, gls):
        if gls in self.gls2lem:
            lem_list = []
            for lemma in self.gls2lem[gls]:
                lem_list.append(lemma)
            return lem_list
        else:
            return []

    def lemmatize(self, sentence):
        lemmas = dict()
        try:
            processed = self.lemmatizer.annotate(sentence)
            lemmatized = processed.sentence[0]
            for token in lemmatized.token:
                lemmas[token.originalText] = token.lemma
        except:
            self.lemmatizer = None
            time.sleep(5)
            self.lemmatizer = corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma".split(), timeout=30000, memory='8G')
        return lemmas

    def rm_igt(self, igt):
        index = -1
        if igt.id in self.igt_ids:
            index = self.igt_ids.index(igt.id)
            self.igt_ids.pop(index)
            labels = self.set_labels.pop(index)
            features = self.set_features.pop(index)
            for i in range(len(labels)):
                source = features[i]['tw_i']
                sys.stderr.write("Removing mapping:" + labels[i] + " - " + source + "\n")
                super()._rm_mapping(self.labels2feats, labels[i], source, igt.id)
                sys.stderr.write("Removing mapping:" + source + " - " + labels[i] + "\n")
                super()._rm_mapping(self.feats2labels, source, labels[i], igt.id)
        return index

    def add_igt(self, igt):
        igt_labels, igt_features = [], []
        gls_src_mapping, src_gls_mapping = [], []
        a_tier, twds_tier = {}, {}
        if igt.get('a') != None:
            for item in igt.get('a').items:
                a_tier[item.get_attribute('source')] = item.get_attribute('target')
        elif igt.get('bilingual-alignments') != None:
            for item in igt.get('bilingual-alignments').items:
                a_tier[item.get_attribute('source')] = item.get_attribute('target')
        elif igt.get('bilingual-alignments_a') != None:
            for item in igt.get('bilingual-alignments_a').items:
                a_tier[item.get_attribute('source')] = item.get_attribute('target')

        if igt.get('tw-ds') != None:
            for item in igt.get('tw-ds').items:
                twds_tier[item.get_attribute('dep')] =  item.id

        lemmas = self.lemmatize(self.get_translation(igt))
        if "n't" not in lemmas:
            lemmas["n't"] = "not"

        for item in igt.get('tw').items:
            src = self.get_value(igt, 'tw', item.id)
            gls = '==NA=='
            if item.id in a_tier:
                gls_id = a_tier[item.id]
                if igt.get('g').get(gls_id):
                    gls = self.get_value(igt, 'g', gls_id)
            if src in string.punctuation:
                gls = '==PUNC=='

            self.curr_map = {}

            # current translation word
            self.curr_map['tw_i'] = src

            # current lemma
            lemma = '==UNK=='
            if src in lemmas:
                lemma = lemmas[src]
            elif "n't" in src:
                index = src.find("n't")
                if src[index:] in lemmas and src[:index] in lemmas:
                    lemma = lemmas[src[:index]] + lemmas[src[index:]]
            elif src in string.punctuation:
                lemma = src

            self.curr_map['lemma_i'] = lemma

            # current tw part of speech
            twps = '==UNK=='
            if igt.get('tw-ps') != None:
                twps_id = igt.referrers(item.id)['alignment'][0]
                if twps_id:
                    twps_idx = re.search(r'(\d+)$', twps_id).group(1)
                    twps_tier = twps_id.split(twps_idx)[0]
                    if igt.get(twps_tier) != None:
                        twps = self.get_value(igt, twps_tier, twps_id)
                    else:
                        twps = self.get_value(igt, 'tw-ps', twps_id)
            elif igt.get('tw-pos') != None:
                twps_id = igt.referrers(item.id)['alignment'][0]
                if twps_id:
                    twps_idx = re.search(r'(\d+)$', twps_id).group(1)
                    twps_tier = twps_id.split(twps_idx)[0]
                    if igt.get(twps_tier) != None:
                        twps = self.get_value(igt, twps_tier, twps_id)
                    else:
                        twps = self.get_value(igt, 'tw-pos', twps_id)

            self.curr_map['twps_i'] = twps

            # current tw dependency structure
            twds = '==UNK=='
            if twds_tier and item.id in twds_tier:
                twds = self.get_value(igt, 'tw-ds', twds_tier[item.id])
            self.curr_map['twds_i'] = twds

            igt_features.append(self.curr_map)
            gls_idx = len(igt_labels)
            igt_labels.append(gls)
            gls_src_mapping.append((gls, src, (igt.id, gls_idx)))
            src_gls_mapping.append((src, gls, (igt.id, gls_idx)))
            self.curr_map = None
        self.igt_ids.append(igt.id)
        for mapping in gls_src_mapping:
            self.add_mapping(self.labels2feats, mapping[0], mapping[1], mapping[2])
        for mapping in src_gls_mapping:
            self.add_mapping(self.feats2labels, mapping[0], mapping[1], mapping[2])
        self.set_labels.append(igt_labels)
        self.set_features.append(igt_features)

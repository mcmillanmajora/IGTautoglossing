# -*- coding: utf-8 -*-
"""

@author: aymm
"""
import json
import os
import sys
import string
from FeatureMap import SrcGls_Map, TwGls_Map
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import joblib
from unidecode import unidecode


class IgtGlosser:
    def __init__(self, trainingalg='l2sgd'):
        self.xc_igts = None
        self._igt2index, self._index2igt = dict(), dict()
        self._train_indices, self._test_indices = [], []
        self._sg_crfmodel = sklearn_crfsuite.CRF(
            algorithm=trainingalg,
            c2=0.1,
            max_iterations=50,
            all_possible_transitions=True
            )
        self._tg_crfmodel = sklearn_crfsuite.CRF(
            algorithm=trainingalg,
            c2=0.1,
            max_iterations=50,
            all_possible_transitions=True
            )
        self._sg_map, self._tg_map = None, None
        self._tr_sg_preds, self._test_sg_preds = None, None
        self._tr_tg_preds, self._test_tg_preds = None, None
        self._tr_final_preds, self._test_final_preds = None, None
        self._gram_dict = None
        self._gram_set = None

    @property
    def size(self):
        return len(self.xc_igts)

    def load_corpus(self, xigtcorpus):
        self.xc_igts = xigtcorpus.igts
        sys.stderr.write("Corpus size t=0: " + str(len(self.xc_igts)) + "\n")
        nonempty_igt = set()
        aligned_igt = set()
        removed = []
        for igt in self.xc_igts:
            if igt.get('g') == None:
                sys.stderr.write("IGT " + igt.id + " HAS NO GLOSS TIER...SKIPPING...\n")
                removed.append(igt)
            elif igt.get('m') == None:
                sys.stderr.write("IGT " + igt.id + " HAS NO MORPHEME TIER...SKIPPING...\n")
                removed.append(igt)
            elif igt.get('p') == None:
                sys.stderr.write("IGT " + igt.id + " HAS NO PHRASES TIER...SKIPPING...\n")
                removed.append(igt)
            else:
                sentence = self.get_value(igt, 'p', 'p1')
                if sentence not in nonempty_igt:
                    nonempty_igt.add(sentence)
                    if igt.get('a') != None:
                        aligned_igt.add(igt.id)
                    elif igt.get('bilingual-alignments') != None:
                        aligned_igt.add(igt.id)
                    elif igt.get('bilingual-alignments_a') != None:
                        aligned_igt.add(igt.id)
                    else:
                        removed.append(igt)
                        sys.stderr.write("IGT " + igt.id + " HAS NO ALIGNMENT...\n")
                else:
                    removed.append(igt)
                    sys.stderr.write("IGT " + igt.id + " PREVIOUSLY SEEN PHRASE LINE...\n")
        for igt in removed:
            self.xc_igts.remove(igt)
        self._size = len(self.xc_igts)
        sys.stderr.write("Xigtcorpus size t=1: " + str(len(xigtcorpus)) + "\n")
        sys.stderr.write("List size t=1: " + str(len(self.xc_igts)) + "\n")
        sys.stderr.write("Removed " + str(len(removed)) + " IGT\n")
        sys.stderr.write("Aligned size: " + str(len(aligned_igt)) + "\n")

    def save_sets(self, outfile):
        if self._data["tr_in"]:
            with open(outfile, 'w') as jsonfile:
                json.dump(self._data, jsonfile)
        else:
            raise ValueError("Sets not defined.")

    def _split_sets(self, percent=None, testset=None, trainset=None):
        sys.stderr.write("SIZE = " + str(self.size) + "\n")
        for index in range(len(self.xc_igts)):
            igtid = self.xc_igts[index].id
            self._index2igt[index] = igtid
            self._igt2index[igtid] = index
            index += 1
        if testset:
            self._test_indices = []
            for id in testset:
                if id in self._igt2index:
                    self._test_indices.append(self._igt2index[id])
                else:
                    sys.stderr.write("COULD NOT ADD IGT " + id + " TO TESTSET\n")
            curr_len = len(self._test_indices)
            if trainset:
                self._train_indices = []
                for id in trainset:
                    if id in self._igt2index:
                        self._train_indices.append(self._igt2index[id])
                    else:
                        sys.stderr.write("COULD NOT ADD IGT " + id + " TO TRAINSET\n")
            elif percent:
                split = int(round(self.size * percent))
                test_size = self.size - split
                if curr_len > test_size:
                    raise ValueError("Given testset is greater than given percentage.")
                else:
                    curr_index = self.size - 1
                    while curr_len < test_size:
                        if curr_index not in self._test_indices:
                            self.testset.append(curr_index)
                            curr_index -= 1
                            curr_len += 1
                self._train_indices = [x for x in self._index2igt if x not in self._test_indices]
        else:
            if not percent:
                raise ValueError("Percent must be given if testset is not specified.")
            split = int(round(self.size * percent))
            test_size = self.size - split
            self._train_indices = [x for x in range(split)]
            self._test_indices = [(x + split) for x in range(test_size)]
        sys.stderr.write("Trainset size: " + str(len(self._train_indices)) + "\n")
        sys.stderr.write("Testset size: " + str(len(self._test_indices)) + "\n")
        uni_union = set(self._train_indices + self._test_indices)
        sys.stderr.write("Unique union size: " + str(len(uni_union)) + "\n")

    def get_data(self, percent=None, testset=None, trainset=None):
        self._sg_map = SrcGls_Map(glosser=self)
        self._tg_map = TwGls_Map(glosser=self)
        rm_list = []
        for igt in self.xc_igts:
            try:
                self._sg_map.add_igt(igt)
                self._tg_map.add_igt(igt)
            except:
                rm_list.append(igt)
                sys.stderr.write("Key error at IGT " + igt.id + "\t" + str(sys.exc_info()[0]) + "\n")
        if rm_list:
            for igt in rm_list:
                self.xc_igts.remove(igt)
                sys.stderr.write("Popped IGT at index " + str(self._sg_map.rm_igt(igt)) + "\n")
                sys.stderr.write("Popped IGT at index " + str(self._tg_map.rm_igt(igt)) + "\n")
        # TESTING ID MATCH
        for index in range(len(self.xc_igts)):
            if self.xc_igts[index].id != self._sg_map.igt_ids[index]:
                sys.stderr.write("SG MAP AND XC IGTS ARE MISMATCHED AT INDEX " + str(index) + "\n")
                sys.stderr.write("SG MAP: " + self._sg_map.igt_ids[index] + "\n")
                sys.stderr.write("XC IGT: " + self.xc_igts[index].id + "\n")
            if self.xc_igts[index].id != self._tg_map.igt_ids[index]:
                sys.stderr.write("TG MAP AND XC IGTS ARE MISMATCHED AT INDEX " + str(index) + "\n")
                sys.stderr.write("TG MAP: " + self._tg_map.igt_ids[index]) + "\n"
                sys.stderr.write("XC IGT: " + self.xc_igts[index].id + "\n")
        self._split_sets(percent, testset, trainset)

    def fit_models(self):
        if self._sg_map:
            features = self._sg_map.get_feats(self._train_indices)
            labels = self._sg_map.get_labels(self._train_indices)
            self._sg_crfmodel.fit(features, labels)
        if self._tg_map:
            features = self._tg_map.get_feats(self._train_indices)
            labels = self._tg_map.get_labels(self._train_indices)
            self._tg_crfmodel.fit(features, labels)

    def normalize_maps(self):
        try:
            self._sg_map.normalize()
            self._tg_map.normalize()
        except:
            raise ValueError("Maps cannot be normalized.")

    def annotate_sets(self):
        train_tg_feats = self._tg_map.get_feats(self._train_indices)
        train_sg_feats = self._sg_map.get_feats(self._train_indices)
        test_tg_feats = self._tg_map.get_feats(self._test_indices)
        test_sg_feats = self._sg_map.get_feats(self._test_indices)

        self._tr_sg_preds = self._sg_crfmodel.predict(train_sg_feats)
        self._tr_tg_preds = self._tg_crfmodel.predict(train_tg_feats)
        self._test_sg_preds = self._sg_crfmodel.predict(test_sg_feats)
        self._test_tg_preds = self._tg_crfmodel.predict(test_tg_feats)

        self._tr_final_preds = []
        self._test_final_preds = []

        for index in range(len(self._train_indices)):
            igt = self.xc_igts[self._train_indices[index]]
            tr_sg_pred = self._tr_sg_preds[index]
            tr_tg_pred = self._tr_tg_preds[index]
            tr_tg_feat = train_tg_feats[index]
            pred = self.annotate_igt(igt, tr_sg_pred, tr_tg_pred, tr_tg_feat)
            self._tr_final_preds.append(pred)

        for index in range(len(self._test_indices)):
            igt = self.xc_igts[self._test_indices[index]]
            test_sg_pred = self._test_sg_preds[index]
            test_tg_pred = self._test_tg_preds[index]
            test_tg_feat = test_tg_feats[index]
            pred = self.annotate_igt(igt, test_sg_pred, test_tg_pred, test_tg_feat)
            self._test_final_preds.append(pred)

    def confirm_pred(self, preds, igt, tier):
        never_seen = [None for x in preds]
        mismatched = [None for x in preds]
        confirmed = [None for x in preds]
        for index in range(len(preds)):
            src = self.get_value(igt, tier, igt.get(tier)[index].id)
            gls = preds[index]
            if tier == "m":
                map = self._sg_map
            else:
                map = self._tg_map

            if map.contains_input(src, self._train_indices):
                if map.contains_match(src, gls, self._train_indices):
                    confirmed[index] = src
                else:
                    mismatched[index] = src
            else:
                never_seen[index] = src
        return never_seen, mismatched, confirmed

    def load_gram_list(self):
        self._gram_dict = dict()
        self._gram_set = set()
        with open("./odin_lexicon.txt") as odin_file:
            for line in odin_file:
                words = line.split()
                gram = words[2].lower()
                domain = words[3].lower()
                if gram != "gram":
                    self._gram_set.add(gram)
                    if domain not in self._gram_dict:
                        self._gram_dict[domain] = set()
                    self._gram_dict[domain].add(gram)

    def annotate_igt(self, igt, sg_preds, tg_preds, t_feats):
        self.load_gram_list()
        sg_nvr, sg_mis, sg_conf = self.confirm_pred(sg_preds, igt, 'm')
        tg_nvr, tg_mis, tg_conf = self.confirm_pred(tg_preds, igt, 'tw')

        alignment = [None for x in tg_preds]
        for index in range(len(tg_conf)):
            if tg_conf[index]:
                # gls changed to lemma; '==NA==' changed to '0'
                gls = tg_preds[index]
                if gls in sg_preds:
                    try:
                        a_index = sg_preds.index(gls)
                        if sg_conf[a_index]:
                            alignment[index] = a_index
                    except:
                        sys.stderr.write("Could not get index for gloss " + gls + " in IGT " + igt.id + "\n")
                if gls == '==NA==':
                    alignment[index] = -1

        final_pred = [None for x in sg_preds]
        for index in range(len(sg_preds)):
            if sg_conf[index]:
                final_pred[index] = sg_preds[index]

        unaligned_tw = [None for x in tg_preds]
        for index in range(len(tg_preds)):
            if not alignment[index]:
                unaligned_tw[index] = t_feats[index]['tw_i']

        # incorrect gloss predicted, but possible seen gloss in translation
        for index in range(len(sg_mis)):
            if sg_mis[index]:
                morph = sg_mis[index]
                glosses = self._sg_map.get_gls_set(morph, self._train_indices)
                for gls in glosses:
                    tg_words = self._tg_map.get_src_set(gls, self._train_indices)
                    for tg in tg_words:
                        if tg in unaligned_tw:
                            final_pred[index] = gls
                            unaligned_tw[unaligned_tw.index(tg)] = None
                            sys.stderr.write("Fixed gloss \"" + gls + "\" in IGT " + igt.id + "\n")

        # get EN info
        ps = [None for x in unaligned_tw]
        ds = [None for x in unaligned_tw]
        lem = [None for x in unaligned_tw]
        for t_index in range(len(unaligned_tw)):
            if unaligned_tw[t_index]:
                lem[t_index] = t_feats[t_index]['lemma_i']
                ps[t_index] = t_feats[t_index]['twps_i']
                ds[t_index] = t_feats[t_index]['twds_i']

        # never before seen source token
        word2morph = dict() # [w1:[0, 1, 2], w2:...]
        for index in range(len(igt.get('m'))):        
            m_id = igt.get('m')[index].id
            w_id = igt.referents(m_id)['segmentation'][0]
            if w_id in word2morph:
                word2morph[w_id].append(index)
            else:
                word2morph[w_id] = [index]

        for index in range(len(sg_nvr)):
            if sg_nvr[index]:
                morph = sg_nvr[index]
                # exact token match
                if morph in tg_nvr:
                    final_pred[index] = morph
                    tg_nvr[tg_nvr.index(morph)] = None
                    sys.stderr.write("Exact match \"" + morph + "\" in IGT " + igt.id + "\n")
                elif morph in unaligned_tw:
                    final_pred[index] = morph
                    unaligned_tw[unaligned_tw.index(morph)] = None
                    sys.stderr.write("Exact match \"" + morph + "\" in IGT " + igt.id + "\n")
                else:
                    # match with unmatched translation word
                    # and add grams from sg_pred
                    pred_domains = []
                    grams = []
                    if "." in sg_preds[index]:
                        labels = sg_preds[index].split(".")
                        for label in labels:
                            if label in self._gram_set:
                                grams.append(label)
                    
                    attached_ind = []
                    for w in word2morph:
                        wm_list = word2morph[w] 
                        if index in wm_list:
                            attached_ind = word2morph[w]
                    for att_ind in attached_ind:
                        pred_attached = final_pred[att_ind]
                        if pred_attached in self._gram_set:
                            grams.append(pred_attached)

                    for domain in self._gram_dict:
                        for gram in grams:
                            if gram in self._gram_dict[domain]:
                                pred_domains.append(domain)

                    heur_lem = ""
                    # Noun
                    if "case" in pred_domains:
                        if "nom" or "erg" in grams:
                            if "nsubj" in ds:
                                unaligned_tw[ds.index("nsubj")] = None
                                heur_lem = self._pop_lem(ds.index("nsubj"), lem, ds, ps)
                        elif "acc" in grams:
                            if "dobj" in ds:
                                unaligned_tw[ds.index("dobj")] = None
                                heur_lem = self._pop_lem(ds.index("dobj"), lem, ds, ps)
                        elif "dat" in grams:
                            if "iobj" in ds:
                                unaligned_tw[ds.index("iobj")] = None
                                heur_lem = self._pop_lem(ds.index("iobj"), lem, ds, ps)
                        elif "loc" in grams:
                            if "pobj" in ds:
                                unaligned_tw[ds.index("pobj")] = None
                                heur_lem = self._pop_lem(ds.index("pobj"), lem, ds, ps)
                        elif "gen" in grams:
                            if "poss" in ds:
                                unaligned_tw[ds.index("poss")] = None
                                heur_lem = self._pop_lem(ds.index("poss"), lem, ds, ps)
                            elif "possessive" in ds:
                                unaligned_tw[ds.index("possessive")] = None
                                heur_lem = self._pop_lem(ds.index("possessive"), lem, ds, ps)
                        else:
                            if "nn" in ps:
                                unaligned_tw[ps.index("nn")] = None
                                heur_lem = self._pop_lem(ps.index("nn"), lem, ds, ps)
                    # Verb
                    elif "tense" or "aspect" in pred_domains:
                        if "root" in ds:
                            unaligned_tw[ds.index("root")] = None
                            heur_lem = self._pop_lem(ds.index("root"), lem, ds, ps)
                        elif "VB" in ps:
                            unaligned_tw[ps.index("VB")] = None
                            heur_lem = self._pop_lem(ps.index("VB"), lem, ds, ps)
                        elif "VBD" in ps:
                            unaligned_tw[ps.index("VBD")] = None
                            heur_lem = self._pop_lem(ps.index("VBD"), lem, ds, ps)
                        elif "VBG" in ps:
                            unaligned_tw[ps.index("VBG")] = None
                            heur_lem = self._pop_lem(ps.index("VBG"), lem, ds, ps)
                        elif "VBN" in ps:
                            unaligned_tw[ps.index("VBN")] = None
                            heur_lem = self._pop_lem(ps.index("VBN"), lem, ds, ps)
                        elif "VBP" in ps:
                            unaligned_tw[ps.index("VBP")] = None
                            heur_lem = self._pop_lem(ps.index("VBP"), lem, ds, ps)
                        elif "VBZ" in ps:
                            unaligned_tw[ps.index("VBZ")] = None
                            heur_lem = self._pop_lem(ps.index("VBZ"), lem, ds, ps)

                    if heur_lem:
                        for gram in grams:
                            heur_lem += "." + gram
                        final_pred[index] = heur_lem
                        sys.stderr.write("HEUR match \"" + heur_lem + "\" in IGT " + igt.id + "\n")

        for index in range(len(final_pred)):
            if not final_pred[index]:
                if sg_preds[index] != "==PUNC==" and sg_preds[index] != "==NONE==":
                    final_pred[index] = sg_preds[index]
                    sys.stderr.write("SG Assigned \"" + final_pred[index] + "\" in IGT " + igt.id + "\n")
                else:
                    for tw_index in range(len(unaligned_tw)):
                        if unaligned_tw[tw_index]:
                            if lem[tw_index]:
                                final_pred[index] = lem[tw_index]
                                lem[tw_index] = None
                                unaligned_tw[tw_index] = None
                            else:
                                final_pred[index] = unaligned_tw[tw_index]
                                unaligned_tw[tw_index] = None
                            sys.stderr.write("TW Assigned \"" + final_pred[index] + "\" in IGT " + igt.id + "\n")
                            break
            if not final_pred[index]:
                final_pred[index] = sg_preds[index]
                sys.stderr.write("Final Assigned \"" + final_pred[index] + "\" in IGT " + igt.id + "\n")
        return final_pred

    def _pop_lem(self, index, lem_list, ds_list, ps_list):
        lemma = lem_list[index]
        lem_list[index] = None
        ds_list[index] = None
        ps_list[index] = None
        return lemma

    def format_igt(self, igt, pred, sg_pred, tg_pred, label, seen_labels):
        source = ""
        for item in igt.get('w'):
            source += self.get_value(igt, 'w', item.id) + " "
        output = "\n\tSrc Line: {0:>10}".format(self.get_value(igt, 'p', 'p1'))
        output += "\n\tSrc Words: {0:>10}".format(source)
        output += "\n\tSrc Seen: {0:>10}".format(str(seen_labels))
        output += "\n\tPredicted from Src: {0:>10}".format(str(sg_pred))
        output += "\n\n\tFinal Prediction: {0:>10}".format(str(pred))
        output += "\n\tGold Label      : {0:>10}".format(str(label))
        output += "\n\n\tPredicted from Trs: {0:>10}".format(str(tg_pred))
        translation = ""
        for item in igt.get('tw'):
            translation += self.get_value(igt, 'tw', item.id) + " "
        output += "\n\tTrs Words: {0:>10}".format(translation)
        output += "\n\tTrs Line: {0:>10}".format(self.get_translation(igt))
        output += "\n"
        return output

    def maps_tostr(self):
        output = "\nSource to Gloss Map\n"
        output += str(self._sg_map)
        output += "Translation to Gloss Map\n"
        output += str(self._tg_map)
        return output

    def _print_err_map(self, err_map):
        output = ""
        if err_map:
            count_map = dict()
            for key in err_map:
                count_map[key] = len(err_map[key])
            count_map = sorted(count_map.items(), key=lambda kv: kv[1], reverse = True)
            for key_count in count_map:
                output += "\tGold Label: " + key_count[0]
                output += "\t<Errors: " + str(key_count[1]) + ">\n"
                output += "\t\tPreds: " + str(err_map[key_count[0]]) + "\n"
        else:
            output += "\t- No Errors -\n"
        return output

    def errors_tostr(self, errs, title):
        output = "\n" + title + "\n"
        output += "--Morpheme Errors\n"
        output += self._print_err_map(errs[0])
        output += "--Stem Errors\n"
        output += self._print_err_map(errs[1])
        output += "--Gram Errors\n"
        output += self._print_err_map(errs[2])
        return output

    def get_accuracies(self):
        output = "\n---Test Set Results---"
        output +="\nTest Instances: " + str(len(self._test_indices)) + "\n"
        test_acc, output, test_errors = self.score_set(self._test_indices, output)
        output += "\n---Training Set Results---"
        output +="\nTraining Instances: " + str(len(self._train_indices)) + "\n"
        train_acc, output, train_errors = self.score_set(self._train_indices, output)
        output += "\n"
        return test_acc, train_acc, output, test_errors, train_errors

    def _format_EA(self, results):
        outstr = "\tSeen--"
        outstr += str(results[1][1]) + "/" + str(results[1][0])
        outstr += "\tOOV--" + str(results[2][1]) + "/" + str(results[2][0])
        return outstr

    def _format_res(self, results):
        outstr = str(results[0]) + "\tCorrect: " + str(results[1])
        outstr += "\tIncorrect: " + str(results[2])
        if len(results) == 4:
            outstr += "\tNone: " + str(results[3])
        return outstr + "\n"

    def _add_igt_res(self, igt_res, total_res, seen, unseen):
        length = len(total_res)
        total_res = [(total_res[i] + igt_res[0][i]) for i in range(length)]
        seen = [(seen[i] + igt_res[1][i]) for i in range(length)]
        unseen = [(unseen[i] + igt_res[2][i]) for i in range(length)]
        return total_res, seen, unseen

    def _add_errs(self, igt_errs, errors):
        for i in range(len(errors)):
            for err in igt_errs[i]:
                pred = err[0]
                gold = err[1]
                if gold in errors[i]:
                    errors[i][gold].append(pred)
                else:
                    errors[i][gold] = [pred]

    def score_set(self, set, output):
        preds, golds, sg_preds, tg_preds = None, None, None, None
        results = ""
        if set == self._test_indices:
            preds = self._test_final_preds
            golds = self._sg_map.get_labels(self._test_indices)
            sg_preds = self._test_sg_preds
            tg_preds = self._test_tg_preds
        else:
            preds = self._tr_final_preds
            golds = self._sg_map.get_labels(self._train_indices)
            sg_preds = self._tr_sg_preds
            tg_preds = self._tr_tg_preds

        # morphs_totals, correct predictions, incorrect predictions
        morph_results = [0, 0, 0]
        morph_seen = [0, 0, 0]
        morph_unseen = [0, 0, 0]

        # gold total, correct predictions, incorrect predictions, no prediction
        stem_results =  [0, 0, 0, 0]
        stem_seen = [0, 0, 0, 0]
        stem_unseen = [0, 0, 0, 0]
        gram_results =  [0, 0, 0, 0]
        gram_seen = [0, 0, 0, 0]
        gram_unseen = [0, 0, 0, 0]
        errors = [dict(), dict(), dict()]

        for index in range(len(set)):
            igt = self.xc_igts[set[index]]
            pred = preds[index]
            gold = golds[index]
            sg_pred = sg_preds[index]
            tg_pred = tg_preds[index]
            m_res, stem_res, gram_res, gold_seen, igt_errs = self.get_igt_score(pred, gold)
            morph_results, morph_seen, morph_unseen = self._add_igt_res(m_res, morph_results, morph_seen, morph_unseen)
            stem_results, stem_seen, stem_unseen = self._add_igt_res(stem_res, stem_results, stem_seen, stem_unseen)
            gram_results, gram_seen, gram_unseen = self._add_igt_res(gram_res, gram_results, gram_seen, gram_unseen)
            self._add_errs(igt_errs, errors)
            results += "\nIGT id: " + igt.id
            results += "\nScores: Morph--" + str(m_res[0][1]) + "/" + str(m_res[0][0])
            results += self._format_EA(m_res)
            results += "\nStems--" + str(stem_res[0][1]) + "/" + str(stem_res[0][0])
            results += "\tno_pred=" + str(stem_res[0][3]) + self._format_EA(stem_res)
            results += "\nGrams--" + str(gram_res[0][1]) + "/" + str(gram_res[0][0])
            results += "\tno_pred=" + str(gram_res[0][3]) + self._format_EA(gram_res)
            results += self.format_igt(igt, pred, sg_pred, tg_pred, gold, gold_seen)
        m_acc = morph_results[1] / (morph_results[0] * 1.0)
        stem_prec = str(stem_results[1] / (stem_results[1] + stem_results[2] * 1.0))
        stem_recl = 0
        if stem_results[0] != 0:
            stem_recl = str(stem_results[1] / stem_results[0] * 1.0)
        gram_prec = str(gram_results[1] / (gram_results[1] + gram_results[2] * 1.0))
        gram_recl = 0
        if gram_results[0] != 0:
            gram_recl = str(gram_results[1] / gram_results[0] * 1.0)
        top = "Morpheme Accuracy: " + str(m_acc) + "\n"
        top += "\tTotal: " + self._format_res(morph_results)
        top += "\tSeen: " + self._format_res(morph_seen)
        top += "\tOOV: " + self._format_res(morph_unseen)
        top += "Stem-- Precision: " + stem_prec + "\tRecall: " + stem_recl + "\n"
        top += "\tTotal: " + self._format_res(stem_results)
        top += "\tSeen: " + self._format_res(stem_seen)
        top += "\tOOV: " + self._format_res(stem_unseen)
        top += "Gram-- Precision: " + gram_prec + "\tRecall: " + gram_recl + "\n"
        top += "\tTotal: " + self._format_res(gram_results)
        top += "\tSeen: " + self._format_res(gram_seen)
        top += "\tOOV: " + self._format_res(gram_unseen)
        final_out = output + top + results
        morph_stem_gram = (morph_results, stem_results, gram_results, morph_seen, morph_unseen)
        return morph_stem_gram, final_out, errors

    def get_igt_score(self, pred, gold):
        # morphs_totals, correct predictions, incorrect predictions
        igt_m = [0, 0, 0]
        seen_m = [0, 0, 0]
        unseen_m = [0, 0, 0]

        # gold total, correct predictions, incorrect predictions, no prediction
        igt_stem =  [0, 0, 0, 0]
        stem_seen = [0, 0, 0, 0]
        stem_unseen = [0, 0, 0, 0]
        igt_gram =  [0, 0, 0, 0]
        gram_seen = [0, 0, 0, 0]
        gram_unseen = [0, 0, 0, 0]
        gold_seen = []
        missed_src = []
        missed_gram = []
        missed_stem = []
        for pred_idx in range(len(pred)):
            # morphemes
            seen = self._sg_map.contains_output(gold[pred_idx], self._train_indices)
            if seen:
                gold_seen.append(1)
            else:
                gold_seen.append(0)
            self._add_count(seen, 0, igt_m, seen_m, unseen_m)
            if pred[pred_idx] == gold[pred_idx]:
                self._add_count(seen, 1, igt_m, seen_m, unseen_m)
            else:
                self._add_count(seen, 2, igt_m, seen_m, unseen_m)
                missed_src.append((pred[pred_idx], gold[pred_idx]))

            # labels (stems and grams)
            gold_labels = gold[pred_idx].split('.')
            pred_labels = pred[pred_idx].split('.')
            matched = []

            for gold_idx in range(len(gold_labels)):
                glabel = gold_labels[gold_idx]
                if glabel in self._gram_set:		# GRAMS
                    self._add_count(seen, 0, igt_gram, gram_seen, gram_unseen)
                    if glabel in pred_labels:				# CORRECT
                        self._add_count(seen, 1, igt_gram, gram_seen, gram_unseen)
                        matched.append(glabel)
                    else:									# INCORRECT
                        if gold_idx >= len(pred_labels):
                            self._add_count(seen, 3, igt_gram, gram_seen, gram_unseen)
                            missed_gram.append(("=NONE=", glabel))
                        else:
                            self._add_count(seen, 2, igt_gram, gram_seen, gram_unseen)
                            missed_gram.append(('.'.join(pred_labels), glabel))
                else:								# STEMS
                    self._add_count(seen, 0, igt_stem, stem_seen, stem_unseen)
                    if glabel in pred_labels:				# CORRECT
                        self._add_count(seen, 1, igt_stem, stem_seen, stem_unseen)
                        matched.append(glabel)
                    if glabel not in matched:				# INCORRECT
                        if gold_idx >= len(pred_labels):
                            self._add_count(seen, 3, igt_stem, stem_seen, stem_unseen)
                            missed_stem.append(("=NONE=", glabel))
                        else:
                            self._add_count(seen, 2, igt_stem, stem_seen, stem_unseen)
                            missed_stem.append(('.'.join(pred_labels), glabel))

            if len(pred_labels) > len(gold_labels):
                for i in range(len(pred_labels)):
                    plabel = pred_labels[i]
                    if plabel not in matched and i >= len(gold_labels):
                        if plabel in self._gram_set:
                            self._add_count(seen, 2, igt_gram, gram_seen, gram_unseen)
                            missed_gram.append((plabel, "=NONE="))
                        else:
                            self._add_count(seen, 2, igt_stem, stem_seen, stem_unseen)
                            missed_stem.append((plabel, "=NONE="))

        m_res = (igt_m, seen_m, unseen_m)
        stem_res = (igt_stem, stem_seen, stem_unseen)
        gram_res = (igt_gram, gram_seen, gram_unseen)
        missed = (missed_src, missed_stem, missed_gram)

        return m_res, stem_res, gram_res, gold_seen, missed

    def _add_count(self, seen, index, total_count, seen_count, unseen_count):
        total_count[index] += 1
        if seen:
            seen_count[index] += 1
        else:
            unseen_count[index] += 1

    def dump_models(self, modelfile_sg, modelfile_tg):
        joblib.dump(self._sg_crfmodel, modelfile_sg)
        joblib.dump(self._tg_crfmodel, modelfile_tg)

    def load_models(self, modelfile_sg, modelfile_tg):
        self._sg_crfmodel = joblib.load(modelfile_sg)
        self._tg_crfmodel = joblib.load(modelfile_tg)

    def dump_preds(self, outfile):
        data = dict()
        for i in range(len(self._test_indices)):
            igt = self.xc_igts[self._test_indices[i]]
            data[igt.id] = self._test_final_preds[i]
        with open(outfile, 'w') as jsonfile:
            json.dump(data, jsonfile)

    def dump(self, outfile):
        data = dict()
        igt_list = []
        for igt in self.xc_igts:
            igt_list.append(igt.id)
        data["xc_igt_ids"] = igt_list
        data["igt2index"] = self._igt2index
        data["index2igt"] = self._index2igt
        data["train_indices"] = self._train_indices
        data["test_indices"] = self._test_indices
        data["gram_set"] = self._gram_set
        data["gram_dict"] = self._gram_dict
        data["sg_map"] = self._sg_map.dump()
        data["tg_map"] = self._tg_map.dump()
        with open(outfile, 'w') as jsonfile:
            json.dump(data, jsonfile)


    def load(self, saved_file, xigtcorpus):
        with open(saved_file) as json_file:
            data = json.load(json_file)
            igt_list = data["xc_igt_ids"]
            self.xc_igts = []
            for igtid in igt_list:
                self.xc_igts.append(xigtcorpus.get(igtid))
            self._igt2index = data["igt2index"]
            self._index2igt = data["index2igt"]
            self._train_indices = data["train_indices"]
            self._test_indices = data["test_indices"]
            self._sg_map = SrcGls_Map(glosser=self)
            self._sg_map.load(data["sg_map"])
            self._tg_map = TwGls_Map(glosser=self)
            self._tg_map.load(data["tg_map"])
            self._gram_set = data["gram_set"]
            self._gram_dict = data["gram_dict"]

    def get_value(self, igt, tier, id):
        value = unidecode(igt.get(tier).get(id).value())
        value = value.lower().strip()
        if value not in string.punctuation:
            value = value.strip(string.punctuation)
        return value

    def get_translation(self, igt):
        if igt.get('t') != None:
            return self.get_value(igt, 't', 't1')
        else:
            return self.get_value(igt, 'n', 'n3')

    def get_igt(self, index):
        if index in self._index2igt:
            return self.xc_igts.get(self._index2igt.get(index))
        else:
            return self.xigtcorpus.get(self._index2igt.get(str(index)))

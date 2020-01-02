# -*- coding: utf-8 -*-
"""

@author: aymm
"""
import os
import sys
import numpy as np
import json
from xigt.codecs import xigtxml
from IgtGlosser import IgtGlosser

os.environ["CORENLP_HOME"] = "./stanford-corenlp-full-2018-10-05/"
xc_file = sys.argv[1]
sys.stderr.write('OPENING FILE ' + xc_file + '...\n')
corpus = xigtxml.load(open(xc_file))
outfile = sys.argv[2]
to_save = outfile.split('.')[1]
datafile = '.' + to_save + '_saved.txt'
sgCRF_file = '.' + to_save + '_sgCRF.joblib'
tgCRF_file = '.' + to_save + '_tgCRF.joblib'
IG = IgtGlosser()
sys.stderr.write('LOADING INSTANCES...\n')
with open(outfile, 'w') as out:
    if len(sys.argv) == 4:
        saved_file = sys.argv[3]
        IG.load(saved_file, corpus)
        IG.load_models(sgCRF_file, tgCRF_file)
    else:
        IG.load_corpus(corpus)
        corpus = None
        sys.stderr.write('GETTING DATA...\n')
        IG.get_data(percent=0.9)
        IG.normalize_maps()
        sys.stderr.write('SAVING DATA...\n')
        IG.dump(datafile)
        sys.stderr.write('FITTING MODELS...\n')
        IG.fit_models()
        IG.dump_models(sgCRF_file, tgCRF_file)
    sys.stderr.write('ANNOTATING...\n')
    IG.annotate_sets()
    test_acc, train_acc, report, test_errors, train_errors = IG.get_accuracies()
    out.write("TEST MORPHEMES: " + str(test_acc[0]) + "\n")
    out.write("TEST STEMS: " + str(test_acc[1]) + "\n")
    out.write("TEST GRAMS: " + str(test_acc[2]) + "\n")
    out.write("TRAIN MORPHEMES: " + str(train_acc[0]) + "\n")
    out.write("TRAIN STEMS: " + str(train_acc[1]) + "\n")
    out.write("TRAIN GRAMS: " + str(train_acc[2]) + "\n")
    out.write(report)
    sys.stderr.write('PRINTING ERRORS...\n')
    out.write(IG.errors_tostr(test_errors, "Test Errors"))
    out.write(IG.errors_tostr(train_errors, "Train Errors"))
    sys.stderr.write('PRINTING MAPS...\n')
    out.write(IG.maps_tostr())
    out.close()




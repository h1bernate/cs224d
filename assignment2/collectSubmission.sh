rm -f nlw.zip
zip -r nlw.zip \
data/* \
data_utils/*.py \
checkpoint \
q1_classifier.py  \
q3_RNNLM.py \
q1_softmax.py \
q2_NER.py \
utils.py \
model.py \
q2_NER_summary.py \
q2_initialization.py \
weights/* \
ptb_rnnlm.weights \
ptb_rnnlm.weights.meta \
q2_test.predicted \
weights/ner.weights \
weights/ner.weights.meta
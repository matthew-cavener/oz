import hug
import hdbscan
import logging
import json

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from collections import defaultdict
from sklearn import mixture
from pandas import DataFrame


logger = logging.getLogger(__name__)

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

# Create graph and finalize (finalizing optional but recommended).
g = tf.Graph()
with g.as_default():
    # We will be feeding 1D tensors of text into the graph.
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module(module_url)
    embedded_text = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Create session and initialize.
session = tf.Session(graph=g)
session.run(init_op)

@hug.get('/health')
def health():
    """checks status of application"""
    return "OK"

# TODO: Make FE for semi-supervised labelling, send that off to nlu server
@hug.post('/label')
def label(body, number_of_intents=None):
    utterances = body['utterances']
    keys = ['text', 'intent', 'confidence']
    common_examples = []
    embeddings = session.run(embedded_text, feed_dict={text_input: utterances}).tolist()

    if number_of_intents == None:
        clusterer = hdbscan.HDBSCAN(
            metric='chebyshev',
            min_cluster_size=5,
            min_samples=2,
            prediction_data=True,
            cluster_selection_method='eom',
            alpha=0.8 # TODO: The docs say this should be left alone, and keep the default of 1, but playing with it seems to help, might be different with real data.
            ).fit(np.inner(embeddings, embeddings))
        cluster_probs = hdbscan.all_points_membership_vectors(clusterer)
        labels = clusterer.labels_
        total_intents = labels.max() + 1
    else:
        clusterer = mixture.GaussianMixture(
            n_components=int(number_of_intents),
            covariance_type='full'
            ).fit(embeddings)
        cluster_probs = clusterer.predict_proba(embeddings)
        labels = clusterer.predict(embeddings)
        total_intents = max(labels) + 1

    labels_strings = list(map(str, labels))
    # create list like: [ [utterance, label ] with strings because json keys must be a string
    values = zip(utterances, labels_strings, cluster_probs)
    for value in values:
        common_examples.append(dict(zip(keys, value)))

    message_groups = defaultdict(list)
    for example in common_examples:
        message_groups[example['intent']].append({
            "phrase": example['text'],
            "confidence": example['confidence']
        })

    unlabeled_messages = labels_strings.count("-1")
    total_messages = len(utterances)
    return {
        "label strings": labels_strings,
        "intents found": total_intents,
        "unlabeled messages": unlabeled_messages,
        "labeled messaged": total_messages - unlabeled_messages,
        "total messages": total_messages,
        "message groups": message_groups
    }

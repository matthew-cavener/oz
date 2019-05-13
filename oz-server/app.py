import hug
import hdbscan
import logging
import json

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from collections import defaultdict
from rasa_nlu import load_data
from rasa_nlu import config
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.model import Trainer
from rasa_nlu.model import Interpreter
from os import remove

builder = ComponentBuilder(use_cache=True)

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

@hug.post('/phrase')
def phrase_embed(phrase):
    embedding = session.run(embedded_text, feed_dict={text_input: [phrase]}).tolist()
    return {phrase: embedding}

@hug.post('/train')
def train(body):
    utterances = body['utterances']
    keys = ['text', 'intent']
    common_examples = []
    embeddings = session.run(embedded_text, feed_dict={text_input: utterances}).tolist()
    clusterer = hdbscan.HDBSCAN(
        metric='euclidean',
        min_cluster_size=5,
        min_samples=2,
        prediction_data=True,
        cluster_selection_method='eom',
        alpha=0.8 # TODO: The docs say this should be left alone, and keep the default of 1, but playing with it seems to help, might be different with real data.
        ).fit(np.inner(embeddings, embeddings))

    # create list like: [ [utterance, label ] with strings because stupid JSON can't handle ints, and rasa requires reading from a file...
    labels_strings = list(map(str, clusterer.labels_))
    values = zip(utterances, labels_strings)
    for value in values:
        common_examples.append(dict(zip(keys, value)))

    message_groups = defaultdict(list)
    for example in common_examples:
        message_groups[example['intent']].append(example['text'])

    # TODO: Probably should split up the embedding/clustering and nlu training. embedding/clustering take a couple of seconds, but nlu trining takes a fair bit longer.
    # Plus the HTTP API is better documented and better supported. Plus microservices!
    common_examples = list(filter(lambda i: i['intent'] != "-1", common_examples))
    with open('training_data.json', 'w') as fp: # FIXME: rasa is dumb, and requires reading from a file. Make this a tempfile, or figure out how to get rasa to accept python dict.
        training_data = {
            "rasa_nlu_data": {
                "common_examples": common_examples,
                "regex_features": [],
                "lookup_tables": [],
                "entity_synonyms": []
            }
        }
        json.dump(training_data, fp)

    trainer = Trainer(config.load("training_config.yml"), builder)
    trainer.train(load_data('training_data.json'))
    model_directory = trainer.persist("./rasa_models/")
    remove('training_data.json')

    return {
        "intents found": clusterer.labels_.max(),
        "unlabeled messages": list(clusterer.labels_).count(-1),
        "total messages": len(utterances),
        "message groups": message_groups
    }


@hug.post('/parse')
def parse(utterance):
    interpreter = Interpreter.load("./rasa_models/default/model_20190513-023015", builder)
    return interpreter.parse(str(utterance))

import hug

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# module_url = "/usr/src/app/oz-server/hub_modules/USELarge3"

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

@hug.post('/intents')
def intent_embed(body):
    utterances = body['utterances']
    embeddings = session.run(embedded_text, feed_dict={text_input: utterances}).tolist()
    intent_embeddings = dict(zip(utterances, embeddings))
    return intent_embeddings



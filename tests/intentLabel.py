import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from sklearn.cluster import OPTICS
import pprint

module_url = "./USELarge3"
embed = hub.Module(module_url)

# # Create graph and finalize (finalizing optional but recommended).
# g = tf.Graph()
# with g.as_default():
#     # We will be feeding 1D tensors of text into the graph.
#     text_input = tf.placeholder(dtype=tf.string, shape=[None])
#     embed = hub.Module(module_url)
#     embedded_text = embed(text_input)
#     init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
# g.finalize()

# # Create session and initialize.
# session = tf.Session(graph=g)
# session.run(init_op)

# def phrase_embed(phrase):
#     embedding = session.run(embedded_text, feed_dict={text_input: [phrase]}).tolist()
#     return embedding

def similarity_matrix(data):
    corpus = data['corpus']
    embeddings = list(corpus.keys())
    return np.inner(embeddings, embeddings)

def get_intents(filename):
    intent = False
    intents = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('##') and not line.startswith('## intent:'):
                intent = False
            else:
                intent = True
            if intent == True and line.startswith('- '):
                intents.append(line.rstrip())
        return intents

def generate_corpus(intents):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(intents))
    corpus = dict(zip(embeddings, intents))
    return corpus


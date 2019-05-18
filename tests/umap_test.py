import matplotlib as mpl
mpl.use('Agg')
import hdbscan
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

body = {
    "utterances": [
        "tell me about your personality",
        "why are you here",
        "talk about yourself",
        "tell me some stuff about you",
        "talk some stuff about yourself",
        "I want to know you better",
        "I want to know more about you",
        "who are you",
        "tell me about yourself",
        "tell me about you",
        "about yourself",
        "describe yourself",
        "introduce yourself",
        "say about you",
        "what are you",
        "define yourself",
        "what is your personality",
        "all about you",
        "tell me your age",
        "what's your age",
        "your age",
        "age of yours",
        "how old are you",
        "I'd like to know your age",
        "are you 21 years old",
        "how old is your platform",
        "you are annoying me so much",
        "you're incredibly annoying",
        "I find you annoying",
        "you are annoying",
        "you're so annoying",
        "how annoying you are",
        "you annoy me",
        "you are annoying me",
        "you are irritating",
        "you are such annoying",
        "you're too annoying",
        "you are very annoying",
        "I want you to answer me",
        "answer",
        "answer my question",
        "answer me",
        "give me an answer",
        "answer the question",
        "can you answer my question",
        "tell me the answer",
        "answer it",
        "give me the answer",
        "I have a question",
        "I want you to answer my question",
        "just answer the question",
        "can you answer me",
        "answers",
        "can you answer a question for me",
        "can you answer",
        "answering questions",
        "I want the answer now",
        "just answer my question",
        "you're not helping me",
        "you are bad",
        "you're very bad",
        "you're really bad",
        "you are useless",
        "you are horrible",
        "you are a waste of time",
        "you are disgusting",
        "you are lame",
        "you are no good",
        "you're bad",
        "you're awful",
        "you are not cool",
        "you are not good",
        "you are so bad",
        "you are so useless",
        "you are terrible",
        "you are totally useless",
        "you are very bad"
    ]
}


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


utterances = body['utterances']
keys = ['text', 'intent', 'confidence']
common_examples = []
embeddings = session.run(embedded_text, feed_dict={text_input: utterances}).tolist()

clusterer = hdbscan.HDBSCAN(
    metric='chebyshev',
    min_cluster_size=5,
    min_samples=2,
    prediction_data=True,
    cluster_selection_method='eom',
    alpha=0.8 # TODO: The docs say this should be left alone, and keep the default of 1, but playing with it seems to help, might be different with real data.
    ).fit(np.inner(embeddings, embeddings))


standard_embedding = umap.UMAP(
    n_neighbors=5,
    min_dist=0.0,
    n_components=2,
    random_state=42,
    ).fit_transform(np.inner(embeddings, embeddings))
print(standard_embedding)

# standard_embedding = umap.UMAP(random_state=42).fit_transform(embeddings)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=clusterer.labels_, s=0.1, cmap='Spectral');


plt.savefig('temp.svg')
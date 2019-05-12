import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import hdbscan
import pprint
import json
# import sqlite3

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(module_url)

def phrase_embed(phrase):
    embedding = session.run(embedded_text, feed_dict={text_input: [phrase]}).tolist()
    return embedding

def similarity_matrix(embeddings):
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
                intents.append(line.rstrip()[2:])
        return intents

def generate_embeddings(intents):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(intents))
    return embeddings

intents = get_intents('smalltalk.md')
with open("utterances.json", "w") as utterances_file:
    json.dump({'utterances': intents}, utterances_file, indent=4, sort_keys=True)

data = generate_embeddings(intents)

# data = similarity_matrix(data)


clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=5)
clusterer.fit(data)

print(clusterer.labels_.max())
print(clusterer.labels_)
print(clusterer.probabilities_)

labelled_data = zip(data, clusterer.labels_, clusterer.probabilities_)

def labelled_intents(labelled_data, intents):
    labelled_intents = {}
    i = 0
    for item in labelled_data:
        try:
            labelled_intents[str(item[1])].append([intents[i], item[2]])
            i += 1
        except KeyError:
            labelled_intents[str(item[1])] = [[intents[i], item[2]]]
            i += 1
    return labelled_intents

labelled_intents = labelled_intents(labelled_data, intents)

pprint.pprint(labelled_intents)

print(len(labelled_intents["-1"]))
print(len(labelled_intents["8"]))

with open("labelled_intents_clus5.json", "w") as labelled_intents_file:
    json.dump(labelled_intents, labelled_intents_file, indent=4, sort_keys=True)


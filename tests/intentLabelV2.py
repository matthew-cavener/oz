import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from pandas import DataFrame
import hdbscan
# import sqlite3

module_url = "./USELarge3"
embed = hub.Module(module_url)

# conn = sqlite3.connect('phrases.db')
# c = conn.cursor()

# c.execute(
#     '''
#     CREATE TABLE phrases (
#         PHRASE          TEXT,
#         EMBEDDING       BLOB,
#         CLUSTER_PROB    BLOB,
#         LABEL           INTEGER,
#         LABEL_PROB      REAL
#     );
#     '''
# )


def similarity_matrix(embeddings):
    return np.inner(embeddings, embeddings)

def get_phrases(filename):
    intent = False
    phrases = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('##') and not line.startswith('## intent:'):
                intent = False
            else:
                intent = True
            if intent == True and line.startswith('- '):
                phrases.append(line.rstrip()[2:])
        return phrases

def generate_embeddings(phrases):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(phrases))
    return embeddings


phrases = get_phrases('smalltalk.md')
embeddings = generate_embeddings(phrases)
data = similarity_matrix(embeddings)
clusterer = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=5, prediction_data=True).fit(data)
cluster_probs = hdbscan.all_points_membership_vectors(clusterer)
labels = clusterer.labels_
label_probs = clusterer.probabilities_

http_data = zip(phrases, labels, label_probs)

print(DataFrame(http_data))
# db_data = zip(phrases, embeddings, cluster_probs, labels, label_probs)

# c.executemany('INSERT INTO phrases VALUES (?,?,?,?,?)', db_data)

# conn.commit()
# conn.close()
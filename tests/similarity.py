import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from pandas import DataFrame
import pprint

module_url = "./USELarge3"
embed = hub.Module(module_url)


messages = [
    # Smartphones
    "I like my phone",
    "My phone is not good.",
    "Your cellphone looks great.",

    # Weather
    "Will it snow tomorrow?",
    "Recently a lot of hurricanes have hit the US",
    "Global warming is real",

    # Food and health
    "An apple a day, keeps the doctors away",
    "Eating strawberries is healthy",
    "Is paleo better than keto?",

    # Asking about age
    "How old are you?",
    "what is your age?"
]


def generate_embeddings(intents):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embeddings = session.run(embed(intents))
    return embeddings


def similarity_matrix(embeddings):
    return np.inner(embeddings, embeddings)


embeddings = generate_embeddings(messages)
similarity_matrix = similarity_matrix(embeddings)
print(DataFrame(similarity_matrix))

new_message = [messages[-1]]

new_embedding = generate_embeddings(new_message)

similarity_vector = np.dot(embeddings, np.transpose(new_embedding))

print(DataFrame(similarity_vector))

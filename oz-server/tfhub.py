import tensorflow_hub as hub

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
import numpy as np
from sklearn.preprocessing import normalize

# initialize the word embeddings
# glove_embedding = WordEmbeddings('glove')
flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')

# initialize the document embeddings, mode = mean
document_embeddings = DocumentPoolEmbeddings([flair_embedding_backward,
                                             flair_embedding_forward],
                                             mode='mean')

# create an example sentence
utterances = [
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
        "you're so annoying"
]


embeddings = []
for utterance in utterances:
    sentence = Sentence(utterance, use_tokenizer=True)
    document_embeddings.embed(sentence)
    embeddings.append(sentence.get_embedding().numpy())

embeddings1 = normalize(embeddings, axis=1, norm='l2')

print(np.inner(embeddings1,embeddings1))
print('\n\n\n=================================\n\n\n')


# sentence = Sentence('The grass is green . And the sky is blue .')

# # embed the sentence with our document embedding
# document_embeddings.embed(sentence)

# # now check out the embedded sentence.
# print(sentence.get_embedding())


# module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# # module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
# embed = hub.Module(module_url)
import tensorflow_hub as hub
import tf_sentencepiece

# module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
module_url = "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"
embed = hub.Module(module_url)
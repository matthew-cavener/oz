import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)
import json

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

intents = get_intents('smalltalk.md')
with open("utterances.json", "w") as utterances_file:
    json.dump({'utterances': intents}, utterances_file, indent=4, sort_keys=True)
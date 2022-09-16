from __future__ import unicode_literals, print_function
import pandas as pd
import jsonlines
import json
import spacy
import os
import plac
import random
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from spacy.training import Example

spacy.cli.download("en_core_web_sm")

# TRAIN_DATA = []
# with jsonlines.open('admin.jsonl') as f:
#     for line in f.iter():
#         ents = [tuple(entity) for entity in line['label']]
#         TRAIN_DATA.append((line['data'], {'entities': ents}))
# print(TRAIN_DATA)

# df = pd.read_csv("annotated.csv")
# text = df['ner'][0]
# annotation = df['label'][0]
# labels = json.loads(annotation)
# print(labels[0])
# with open(filename) as train_data:
# 	train=json.load(train_data)
# TRAIN_DATA = []
# for data in train:
# 	ents=[tuple(entity[:3]) for entity in data['entities']]
# 	TRAIN_DATA.append((data['content'], {'entities':ents}))
# with open('{}'.format(filename.replace('json', 'txt')), 'w') as write:
# 	write.write(str(TRAIN_DATA))
# print("[INFO] Stored the spacy training data and filename is {}".format(filename.replace('json','txt')))

nlp=spacy.load('en_core_web_sm')
print(nlp.pipe_names)

model = None
output_dir = Path("TEST")


def train_spacy(data, iterations):
    TRAIN_DATA = data
    if model is not None:
        nlp = spacy.load(model)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')
        print("Created blank 'en' model")

    if 'ner' not in nlp.pipe_names:
        # ner = nlp.create_pipe('ner')
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in tqdm(TRAIN_DATA):
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], losses=losses, drop=0.2, sgd=optimizer)
                # nlp.update([text], [annotations], drop=0.2, sgd=optimizer, losses=losses)
            print(losses)
    return nlp


# prdnlp = train_spacy(TRAIN_DATA, 100)
#
# for text, _ in TRAIN_DATA:
#     doc = prdnlp(text)
#     print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
#
# if output_dir is not None:
#     output_dir = Path(output_dir)
#     if not output_dir.exists():
#         output_dir.mkdir()
#     prdnlp.to_disk(output_dir)
#     print("Saved model to", output_dir)





model = spacy.load(output_dir)
test_text = "TEST_STRING"
doc = model(test_text)
print(doc)
print(doc.ents)
new_data = {}
for ent in doc.ents:
    print(ent.label_, ent.text)
    if ent.label_ in new_data:
        new_data[str(ent.label_)] = (new_data[str(ent.label_)] + " " + ent.text).replace("\n", " ")
    else:
        new_data[str(ent.label_)] = ent.text
print(new_data)

# new_json_data = deepcopy(new_data)
# print(new_json_data)
# with open("LabReports.json", "a") as file:
#     json.dump(new_data, file, indent=4)

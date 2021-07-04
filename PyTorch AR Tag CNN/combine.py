import json

with open("train1.json") as f1:
    train1 = json.load(f1)

with open("train2.json") as f2:
    train2 = json.load(f2)

with open("train3.json") as f3:
    train3 = json.load(f3)

id_images = train1['images'][-1]['id']
id_annotations = train1['annotations'][-1]['id'] + 1


for e in train2['images']:
    e['id'] += id_images
    train1['images'].append(e)


for e in train2['annotations']:
    e['id'] += id_annotations
    e['image_id'] += id_images
    train1['annotations'].append(e)

id_images2 = train1['images'][-1]['id']
id_annotations2 = train1['annotations'][-1]['id'] + 1

for e in train3['images']:
    e['id'] += id_images2
    train1['images'].append(e)


for e in train3['annotations']:
    e['id'] += id_annotations2
    e['image_id'] += id_images2
    train1['annotations'].append(e)


with open('train.json', 'w') as outfile:
    json.dump(train1, outfile)
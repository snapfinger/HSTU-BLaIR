import html
import re

def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '


def clean_metadata(example):
    meta_text = ''
    features_needed = ['title', 'features', 'categories', 'description']
    for feature in features_needed:
        meta_text += feature_process(example[feature])
    example['cleaned_metadata'] = meta_text
    return example


def filter_items_wo_metadata(example, item2meta):
    if example['parent_asin'] not in item2meta:
        example['history'] = ''
    history = example['history'].split(' ')
    filtered_history = [_ for _ in history if _ in item2meta]
    example['history'] = ' '.join(filtered_history)
    return example


def filter_items_wo_metadata_df(row, item2meta):
    if row['item_id'] not in item2meta:
        return None
    return row


def remap_id(dataset):
    user2id = {}
    id2user = []
    item2id = {}
    id2item = []

    for user_id, item_id in zip(dataset['user_id'], dataset['item_id']):
        if user_id not in user2id:
            user2id[user_id] = len(id2user)
            id2user.append(user_id)

        if item_id not in item2id:
            item2id[item_id] = len(id2item)
            id2item.append(item_id)

    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}

    return data_maps


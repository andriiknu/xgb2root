from xgboost import XGBClassifier
import ROOT
agc_models_dir = './models'

import json
def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
            else:
                yield from item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from item_generator(item, lookup_key)

def tree_depth(json_text):
    json_input = json.loads(json_text)
    return max(list(item_generator(json_input, 'depth'))+[0]) + 1

def get_depthes_list(bst):
    return [tree_depth(x) for x in bst.get_booster().get_dump(dump_format = "json")]

def convert_models():
    model_even = XGBClassifier()
    model_even.load_model(f'{agc_models_dir}/model_even.json')
    print(set(get_depthes_list(model_even)))

    model_odd = XGBClassifier()
    model_odd.load_model(f'{agc_models_dir}/model_odd.json')
    print(set(get_depthes_list(model_odd)))


if __name__ == "__main__":
    convert_models()
import yaml
from parameters import Params, params_from_mapping

def load_presets(path):
    with open(path, 'r') as f:
        doc = yaml.safe_load(f)
    presets = doc["presets"]
    return presets

if __name__ == "__main__":
    params = params_from_mapping(load_presets('./params.yml')["baseline"])
    print(params)


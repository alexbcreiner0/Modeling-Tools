import yaml
import numpy as np
import importlib.util
from copy import deepcopy
from dataclasses import fields, is_dataclass, asdict
from typing import get_origin, get_args
from paths import rpath
# from parameters import Params, params_from_mapping

def load_presets(path):
    try:
        with open(rpath(f"models/{path}/data/params.yml"), 'r') as f:
            doc = yaml.safe_load(f)
        return doc["presets"]
    except (FileNotFoundError, KeyError, TypeError):
        with open(rpath(f'models/{path}/data/extra_data.yml'), 'r') as f:
            default_presets = yaml.safe_load(f)
        _dump_to_yaml(default_presets, path)
        return default_presets

def _dump_to_yaml(presets, path):
    class FlowDumper(yaml.SafeDumper):
        pass

    def _repr_list(dumper, data):
        # always use flow style for lists
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    FlowDumper.add_representer(list, _repr_list)
    FlowDumper.ignore_aliases = lambda *a, **k: True

    text = yaml.dump(
        {"presets": presets},
        Dumper=FlowDumper,
        sort_keys=False,
        indent=2,
        width=88
    )

    with open(f"models/{path}/data/params.yml", "w") as f:
        f.write(text)

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def load_from_path(filepath, thing):
    spec = importlib.util.spec_from_file_location(thing, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # print(f"Attempting to load {thing} from module")
    cls = getattr(module, thing)
    return cls

def coerce_value(val, anno):
    """Best-effort coercion based on the dataclass field annotation."""
    if anno is None:
        return val

    # np.ndarray (common case)
    if anno is np.ndarray or getattr(anno, "__name__", "") == "ndarray":
        return np.asarray(val)

    # typing.Optional[T]
    origin = get_origin(anno)
    if origin is not None:
        args = get_args(anno)
        if origin is list:
            # List[T]
            inner = args[0] if args else None
            return [ coerce_value(v, inner) for v in (val or []) ]
        if origin is tuple:
            inner = args[0] if args else None
            return tuple(coerce_value(v, inner) for v in (val or []))
        if origin is dict:
            k_anno, v_anno = (args + (None, None))[:2]
            return { coerce_value(k, k_anno): coerce_value(v, v_anno) for k, v in (val or {}).items() }
        if origin is type(None):  # Optional[None]? ignore
            return val
        if origin is np.ndarray:  # rare typing usage
            return np.asarray(val)

    # Nested dataclass?
    if is_dataclass(anno):
        # If a nested dataclass appears, instantiate it from the dict
        sub_fields = {f.name: f for f in fields(anno)}
        kwargs = {}
        for k, v in (val or {}).items():
            if k in sub_fields:
                kwargs[k] = coerce_value(v, sub_fields[k].type)
        return anno(**kwargs)

    # Basic scalars
    if anno in (float, int, bool, str):
        try:
            return anno(val)
        except Exception:
            return val  # fall back

    return val  # default: no change

def params_from_mapping(map: dict, dataclass_path: str):
    Params = load_from_path(dataclass_path, "Params")
    
    params_fields = fields(Params)
    kwargs = {}
    for f in params_fields:
        if f.name in map:
            kwargs[f.name] = coerce_value(map[f.name], f.type)

    # field_names = {f.name for f in fields(Params)}
    # filtered = {k: v for k, v in map.items() if k in field_names}
    return Params(**kwargs)

def to_plain(obj): # opaque as fuck chatgpt code for converting the parameters dataclass to a yaml-friendly dictionary
    """Recursively convert dataclass / numpy types to YAML-friendly Python types."""
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def format_plot_config(cfg: dict, commodities: list[str]) -> dict:
    # build a mapping like {"c0": "Corn", "c1": "Iron", ...}
    fmt = {f"c{i}": name for i, name in enumerate(commodities)}

    def rec(x):
        if isinstance(x, str):
            try:
                return x.format(**fmt)
            except Exception:
                return x  # leave untouched if no placeholders match
        if isinstance(x, list):
            return [rec(v) for v in x]
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        return x

    return rec(deepcopy(cfg))

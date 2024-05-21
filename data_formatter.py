import os
import json
import numpy as np
from alive_progress import alive_it
import sys
from copy import deepcopy

JSON_PATH = "./out"
JSON_PATH_FORMATTED = "./out_formatted"


def group_array(array: list[float], g: int):
    out = []
    i = 0
    l = len(array)
    while i < l:
        try:
            v = 0
            for n in range(g):
                v += array[i + n]
            v /= g
            out.append(v)
            i += g
        except:
            return out
    return out


def invlerp(min, max, v):
    return (v - min) / (max - min)


def flatten_dict(d: dict[str, list]):
    out_array = []
    for k in d.keys():
        array = np.array(d[k])
        array = array.flatten()
        out_array += array.tolist()
    return out_array


def load_file(path: str):
    with open(f"{JSON_PATH}/{path}", "r") as f:
        return group_array(flatten_dict(json.load(f)), group)


def default_target():
    for p in alive_it(os.listdir(JSON_PATH)):
        data = load_file(p)
        with open(f"{JSON_PATH_FORMATTED}/{p}", "w+") as f:
            json.dump({"data": data}, f)


def change():
    it = alive_it(
        sorted(os.listdir(JSON_PATH), key=lambda x: x.split("_")[-1].split(".")[0])
    ).__iter__()
    first = next(it)
    biggest = load_file(first)
    smallest = deepcopy(biggest)

    for p in it:
        current_data = load_file(p)
        for i, d in enumerate(current_data):
            if d > biggest[i]:
                biggest[i] = d
            if d < smallest[i]:
                smallest[i] = d

    for p in alive_it(os.listdir(JSON_PATH)):
        data = load_file(p)
        for v in range(len(data)):
            data[v] = invlerp(smallest[v], biggest[v], data[v])
        with open(f"{JSON_PATH_FORMATTED}/{p}", "w+") as f:
            json.dump({"data":data}, f)


group = 0
target = default_target
for a in sys.argv:
    if a.startswith("group="):
        group = int(a.split("=")[1])
    if a == "change":
        target = change

target()

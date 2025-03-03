import json
from enum import Enum
from typing import Type, Any, Dict, List

from pydantic import BaseModel, create_model


def serialize_base_model(obj: Type[BaseModel]) -> str:
    return json.dumps(obj.model_json_schema(), ensure_ascii=False)


def dereference_json_schema(json_schema: Dict["str", Any]) -> Dict["str", Any]:
    model_map = json_schema.get("$defs", {})

    def dereference(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"].split("/")[-1]
                return dereference(model_map[ref])
            else:
                return {k: dereference(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [dereference(x) for x in obj]
        else:
            return obj

    result = {}
    for k, v in json_schema.items():
        if k == "$defs":
            continue

        result[k] = dereference(v)

    return result


def deserialize_base_model(json_schema: Dict[str, Any]) -> Type[BaseModel]:
    fields = {}

    for k, v in dereference_json_schema(json_schema)["properties"].items():

        if "enum" in v:
            dynamic_enum = Enum(v["title"], {x: x for x in v["enum"]})
            fields[k] = (dynamic_enum, ...)

        elif v["type"] == "string":
            fields[k] = (str, ...)

        elif v["type"] == "integer":
            fields[k] = (int, ...)

        elif v["type"] == "number":
            fields[k] = (float, ...)

        elif v["type"] == "boolean":
            fields[k] = (bool, ...)

        elif v["type"] == "object":
            fields[k] = (deserialize_base_model(v), ...)

        elif v["type"] == "array":
            item_type = v["items"]["type"]
            if item_type == "string":
                fields[k] = (List[str], ...)
            elif item_type == "integer":
                fields[k] = (List[int], ...)
            elif item_type == "number":
                fields[k] = (List[float], ...)
            elif item_type == "boolean":
                fields[k] = (List[bool], ...)
            elif item_type == "object":
                fields[k] = (List[deserialize_base_model(v["items"])], ...)

    return create_model(json_schema["title"], **fields)

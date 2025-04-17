from enum import Enum
from typing import Any, Dict, List, Type

from pydantic import BaseModel, create_model

__all__ = ["deserialize_base_model", "serialize_base_model"]


def serialize_base_model(obj: Type[BaseModel]) -> Dict[str, Any]:
    return obj.model_json_schema()


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


def parse_field(v: Dict[str, Any]) -> Any:
    t = v["type"]
    if t == "string":
        return str
    elif t == "integer":
        return int
    elif t == "number":
        return float
    elif t == "boolean":
        return bool
    elif t == "object":
        return deserialize_base_model(v)
    elif t == "array":
        inner_type = parse_field(v["items"])
        return List[inner_type]
    else:
        raise ValueError(f"Unsupported type: {t}")


def deserialize_base_model(json_schema: Dict[str, Any]) -> Type[BaseModel]:
    fields = {}
    properties = dereference_json_schema(json_schema).get("properties", {})

    for k, v in properties.items():
        if "enum" in v:
            dynamic_enum = Enum(v["title"], {x: x for x in v["enum"]})
            fields[k] = (dynamic_enum, ...)
        else:
            fields[k] = (parse_field(v), ...)
    return create_model(json_schema["title"], **fields)

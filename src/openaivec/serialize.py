from enum import Enum
from typing import Any, Dict, List, Type, Literal

from pydantic import BaseModel, Field, create_model

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
        # Check if it's a generic object (dict) or a nested model
        if "properties" in v:
            return deserialize_base_model(v)
        else:
            return dict
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
            enum_values = v["enum"]
            
            # Try to create a standard Enum first (for compatibility)
            try:
                dynamic_enum = Enum(v["title"], {x: x for x in enum_values})
                description = v.get("description")
                default_value = v.get("default")
                
                if default_value is not None:
                    field_info = Field(default=default_value, description=description) if description is not None else Field(default=default_value)
                else:
                    field_info = Field(description=description) if description is not None else Field()
                
                fields[k] = (dynamic_enum, field_info)
            except (ValueError, TypeError):
                # If Enum creation fails (e.g., mixed types), use Literal
                # Create a Union of Literal types for each value
                if len(enum_values) == 1:
                    literal_type = Literal[enum_values[0]]
                else:
                    # Create Literal with multiple values
                    literal_type = Literal[tuple(enum_values)]
                
                description = v.get("description")
                default_value = v.get("default")
                
                if default_value is not None:
                    field_info = Field(default=default_value, description=description) if description is not None else Field(default=default_value)
                else:
                    field_info = Field(description=description) if description is not None else Field()
                
                fields[k] = (literal_type, field_info)
        else:
            description = v.get("description")
            default_value = v.get("default")
            
            if default_value is not None:
                field_info = Field(default=default_value, description=description) if description is not None else Field(default=default_value)
            else:
                field_info = Field(description=description) if description is not None else Field()
            
            fields[k] = (parse_field(v), field_info)
    return create_model(json_schema["title"], **fields)

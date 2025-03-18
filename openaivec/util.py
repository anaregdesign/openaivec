import functools
import time
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import chain
from typing import Callable, List, Optional, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel
from pyspark.sql.types import ArrayType, BooleanType, FloatType, IntegerType, StringType, StructField, StructType

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def split_to_minibatch(b: List[T], batch_size: int) -> List[List[T]]:
    """Splits the list into sublists of size `batch_size`."""
    return [b[i : i + batch_size] for i in range(0, len(b), batch_size)]


def map_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Splits the list `b` into batches of size `batch_size` and applies the function `f` to each batch.
    The results (each a list) are then flattened into a single list.
    """
    batches = split_to_minibatch(b, batch_size)
    return list(chain.from_iterable(f(batch) for batch in batches))


def map_minibatch_parallel(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Splits the list `b` into batches of size `batch_size` and applies the function `f` to each batch.
    The results (each a list) are then flattened into a single list.
    This version uses parallel processing to apply the function to each batch.
    """
    batches = split_to_minibatch(b, batch_size)
    with ThreadPoolExecutor() as executor:
        results = executor.map(f, batches)
    return list(chain.from_iterable(results))


def map_unique(b: List[T], f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Applies the function `f` only once to the unique values in the list `b` (preserving their order),
    and then maps the resulting values back to match the original list.
    This avoids repeated execution of `f` for duplicate values.
    """
    # Use dict.fromkeys to remove duplicates while preserving the order
    unique_values = list(dict.fromkeys(b))
    value_to_index = {v: i for i, v in enumerate(unique_values)}
    results = f(unique_values)
    return [results[value_to_index[value]] for value in b]


def map_unique_minibatch(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Uses minibatch processing on the unique values of the list `b`.
    The function `f` is applied to these unique values in batches,
    and the results are mapped back to match the order of the original list.
    """
    return map_unique(b, lambda x: map_minibatch(x, batch_size, f))


def map_unique_minibatch_parallel(b: List[T], batch_size: int, f: Callable[[List[T]], List[U]]) -> List[U]:
    """
    Uses minibatch processing on the unique values of the list `b`.
    The function `f` is applied to these unique values in batches using parallel processing,
    and the results are mapped back to match the order of the original list.
    """
    return map_unique(b, lambda x: map_minibatch_parallel(x, batch_size, f))


def python_type_to_spark(python_type):
    origin = get_origin(python_type)

    # For list types (e.g., List[int])
    if origin is list or origin is List:
        # Retrieve the inner type and recursively convert it
        inner_type = get_args(python_type)[0]
        return ArrayType(python_type_to_spark(inner_type))

    # For Optional types (Union[..., None])
    elif origin is Union:
        non_none_args = [arg for arg in get_args(python_type) if arg is not type(None)]
        if len(non_none_args) == 1:
            return python_type_to_spark(non_none_args[0])
        else:
            raise ValueError(f"Unsupported Union type with multiple non-None types: {python_type}")

    # For nested Pydantic models (to be treated as Structs)
    elif isinstance(python_type, type) and issubclass(python_type, BaseModel):
        return pydantic_to_spark_schema(python_type)

    # Basic type mapping
    elif python_type is int:
        return IntegerType()
    elif python_type is float:
        return FloatType()
    elif python_type is str:
        return StringType()
    elif python_type is bool:
        return BooleanType()
    else:
        raise ValueError(f"Unsupported type: {python_type}")


def pydantic_to_spark_schema(model: Type[BaseModel]) -> StructType:
    fields = []
    for field_name, field in model.model_fields.items():
        field_type = field.annotation
        # Use outer_type_ to correctly handle types like Optional
        spark_type = python_type_to_spark(field_type)
        # Set nullable to True (adjust logic as needed)
        fields.append(StructField(field_name, spark_type, nullable=True))
    return StructType(fields)


def backoff(
    exception: Exception, interval: Optional[int] = None, max_retries: Optional[int] = None
) -> Callable[..., V]:
    def decorator(func: Callable[..., V]) -> Callable[..., V]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> V:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exception:
                    attempt += 1
                    if max_retries is not None and attempt >= max_retries:
                        raise
                    if interval is not None:
                        time.sleep(interval)

        return wrapper

    return decorator

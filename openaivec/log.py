import functools
import json
import time
import uuid
from logging import Logger
from typing import Callable

__ALL__ = ["observe"]


def observe(logger: Logger):
    """
    Decorator factory that logs the start and end of a method call.

    Args:
        logger (Logger): Logger instance used for logging.

    Returns:
        Callable: A decorator that wraps a method with logging.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def decorated(self: object, *args, **kwargs):
            """
            Wrapped method that logs the start and end of the method call.

            Args:
                self (object): Instance of the class.
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                Any: The result from the original method call.
            """
            l: Logger = logger.getChild(self.__class__.__name__).getChild(func.__name__)
            transaction_id: str = str(uuid.uuid4())
            l.info(
                json.dumps(
                    {
                        "transaction_id": transaction_id,
                        "type": "start",
                        "class": self.__class__.__name__,
                        "method": func.__name__,
                        "logged_at": time.time_ns(),
                    }
                )
            )
            try:
                res = func(self, *args, **kwargs)

            finally:
                l.info(
                    json.dumps(
                        {
                            "transaction_id": transaction_id,
                            "type": "end",
                            "class": self.__class__.__name__,
                            "method": func.__name__,
                            "logged_at": time.time_ns(),
                        }
                    )
                )

            return res

        return decorated

    return decorator

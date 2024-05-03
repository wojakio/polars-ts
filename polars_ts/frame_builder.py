from typing import Any, Dict, Generic, Tuple

import polars as pl

from .types import FrameType


class FrameBuilder(Generic[FrameType]):
    def __init__(self, **param: Any):
        self._param: Dict[str, Any] = param

    def to_frame(self) -> FrameType:
        result = pl.DataFrame().with_columns(
            [
                pl.lit(value).alias(name)
                for name, value in self._param.items()
            ]
        )

        return result

    def __repr__(self) -> str:
        rep = str(self._param)
        # required = ",".join(
        #     sorted([f"{name}:{dtype}" for name, dtype in self._required_schema.items()])
        # )

        # optionals = ",".join(
        #     sorted(
        #         [
        #             f"{name}:{dtype}={self._defaults[name]}"
        #             for name, dtype in self._optional_schema.items()
        #         ]
        #     )
        # )

        # rep = f"{self.__class__}({required})[{optionals}]"

        return rep

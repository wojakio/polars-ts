from typing import Any, Dict, List, Tuple

import polars as pl
from polars.type_aliases import JoinStrategy

from .grouper import Grouper
from .types import FrameType


class WithParams:
    def __init__(self, **required: Any):
        self._required_schema: Dict[str, pl.DataType] = required
        self._optional_schema: Dict[str, pl.DataType] = dict()
        self._defaults: Dict[str, Any] = dict()

    def optional(self, **kwargs: Any) -> "WithParams":
        bad_params = set(self._required_schema.keys()).intersection(kwargs.keys())
        if len(bad_params) > 0:
            raise ValueError(f"Params cannot be required and optional: {bad_params}")

        self._optional_schema.update(kwargs)
        return self

    def defaults(self, **kwargs: Any) -> "WithParams":
        disallowed_defaults = set(self._required_schema.keys()).intersection(
            kwargs.keys()
        )
        if len(disallowed_defaults) > 0:
            raise ValueError(
                f"Cannot set default on a required parameter: {disallowed_defaults}"
            )

        # missing_defaults
        missing_defaults = set(self._optional_schema.keys()).difference(kwargs.keys())
        if len(missing_defaults) > 0:
            raise ValueError(
                f"Optional params need default values. Missing: {missing_defaults}"
            )

        # unknown params
        unknown_params = set(kwargs.keys()).difference(self._optional_schema.keys())
        if len(unknown_params) > 0:
            raise ValueError(
                f"Params must be declared before a default is given: {unknown_params}"
            )

        self._defaults.update(kwargs)
        return self

    @staticmethod
    def _set_type(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
        if dtype == pl.Categorical:
            expr = expr.cast(pl.String)

        return expr.cast(dtype)

    def names(self) -> List[str]:
        all_names = set(self._required_schema.keys()).union(
            self._optional_schema.keys()
        )
        return sorted(all_names)

    def apply(self, df: FrameType, params: FrameType) -> Tuple[FrameType, List[str]]:
        missing_required_params = set(self._required_schema.keys()).difference(
            params.columns
        )
        if len(missing_required_params) > 0:
            raise ValueError(f"Missing required parameters: {missing_required_params}")

        missing_optional_params = set(self._optional_schema.keys()).difference(
            params.columns
        )
        if len(missing_optional_params) > 0:
            params = params.with_columns(
                [pl.lit(self._defaults[k]).alias(k) for k in missing_optional_params]
            )

        full_schema = self._optional_schema | self._required_schema

        mismatch_param_dtype = [
            (name, dtype, full_schema[name])
            for name, dtype in df.schema.items()
            if name in full_schema and dtype != full_schema[name]
        ]

        if len(mismatch_param_dtype) > 0:
            raise ValueError(
                f"Type-mismatch on input frame columns: {mismatch_param_dtype}"
            )

        params = params.with_columns(
            [
                WithParams._set_type(pl.col(name), dtype)
                for name, dtype in full_schema.items()
            ]
        )

        common_cols = Grouper.common_categories(df, params)
        join_type: JoinStrategy = "cross" if len(common_cols) == 0 else "left"
        df_with_params = df.join(params, on=common_cols, how=join_type)

        result_cols = (
            pl.Series(
                values=df.columns + Grouper.categories(params, include_time=False)
            )
            .unique(maintain_order=True)
            .to_list()
        )

        return df_with_params, result_cols

    def __repr__(self) -> str:
        required = ",".join(
            sorted([f"{name}:{dtype}" for name, dtype in self._required_schema.items()])
        )

        optionals = ",".join(
            sorted(
                [
                    f"{name}:{dtype}={self._defaults[name]}"
                    for name, dtype in self._optional_schema.items()
                ]
            )
        )

        rep = f"WithParams({required})[{optionals}]"

        return rep

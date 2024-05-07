from typing import Any, Dict, List, Tuple

import polars as pl
from polars.type_aliases import JoinStrategy

from .grouper import Grouper
from .types import cast_dtype, FrameType


class ParamSchema:
    def __init__(self, **required: Any):
        self._required_schema: Dict[str, pl.DataType] = required
        self._optional_schema: Dict[str, pl.DataType] = dict()
        self._defaults: Dict[str, Any] = dict()
        self._allow_additional_params: bool = False

    def allow_additional_params(self) -> "ParamSchema":
        self._allow_additional_params = True
        return self

    def optional(self, **kwargs: Any) -> "ParamSchema":
        bad_params = set(self._required_schema.keys()).intersection(kwargs.keys())
        if len(bad_params) > 0:
            raise ValueError(f"Params cannot be required and optional: {bad_params}")

        self._optional_schema.update(kwargs)
        return self

    def defaults(self, **kwargs: Any) -> "ParamSchema":
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

    def names(self) -> List[str]:
        all_names = set(self._required_schema.keys()).union(
            self._optional_schema.keys()
        )
        return sorted(all_names)

    def validate(self, df: FrameType, params: FrameType) -> Dict[str, pl.DataType]:
        missing_required_params = set(self._required_schema.keys()).difference(
            params.columns
        )
        if len(missing_required_params) > 0:
            raise ValueError(f"Missing required parameters: {missing_required_params}")

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

        return full_schema

    def apply(self, df: FrameType, params: FrameType) -> Tuple[FrameType, List[str]]:
        full_schema = self.validate(df, params)

        # add optional parameters that weren't supplied in params
        missing_optional_params = set(self._optional_schema.keys()).difference(
            params.columns
        )

        if len(missing_optional_params) > 0:
            params = params.with_columns(
                [pl.lit(self._defaults[k]).alias(k) for k in missing_optional_params]
            )

        # attempt to correct dtypes of parameters
        params = params.with_columns(
            [cast_dtype(pl.col(name), dtype) for name, dtype in full_schema.items()]
        )

        common_cols = Grouper.common_categories(df, params)
        join_type: JoinStrategy = "cross" if len(common_cols) == 0 else "left"
        df_with_params = df.join(params, on=common_cols, how=join_type).with_columns(
            [
                pl.col(param_name).fill_null(param_default_value)
                for param_name, param_default_value in self._defaults.items()
                if param_default_value is not None
            ]
        )

        param_cols = [
            p
            for p in Grouper.columns(params, include_time=False)
            if p not in full_schema.keys()
        ]
        result_cols = (
            pl.Series(values=df.columns + param_cols)
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

        rep = f"{self.__class__}({required})[{optionals}]"

        return rep

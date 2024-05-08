from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set

import polars as pl
from polars.type_aliases import JoinStrategy

from .grouper import Grouper
from .types import cast_dtype, FrameType


@dataclass
class Param:
    group_key: str
    name: str
    dtype: pl.DataType
    default: Optional[Any] = None


class ParamSchema:
    def __init__(self, params: List[Any]):
        self._params: Dict[str, List[Param]] = dict()

        seen = set()
        for param_spec in params:
            param = Param(*param_spec)
            unique_key = f"{param.group_key}_{param.name}"
            if unique_key in seen:
                raise ValueError(f"Encountered duplicate param spec {unique_key}")

            if param.group_key not in self._params:
                self._params[param.group_key] = []

            self._params[param.group_key].append(param)
            seen.add(unique_key)

    # def __init__(self, **required: Any):
    #     self._required_schema: Dict[str, pl.DataType] = required
    #     self._optional_schema: Dict[str, Tuple[Any, pl.DataType]] = dict()

    # def optional(self, name: str, **kwargs: Any) -> "ParamSchema":
    #     bad_params = set(self._required_schema.keys()).intersection(kwargs.keys())
    #     if len(bad_params) > 0:
    #         raise ValueError(f"Params cannot be required and optional: {bad_params}")

    #     self._optional_schema.update(kwargs)
    #     return self

    def _groups(self, group_key: str, invert: bool) -> List[str]:
        all_groups = set(self._params.keys())
        gs = all_groups if group_key == "*" else {group_key}
        if invert:
            gs = all_groups.difference(gs)

        return sorted(gs)

    def params(self, group_key: str, invert: bool) -> List[Param]:
        result: List[Param] = []
        for g in self._groups(group_key, invert):
            result = result + self._params[g]

        return result

    def names(self, group_key: str, invert: bool) -> List[str]:
        result: Set[str] = set()
        for g in self._groups(group_key, invert):
            result = result | {p.name for p in self._params[g]}

        return sorted(result)

        # all_names = set(self._required_schema.keys()).union(
        #     self._optional_schema.keys()
        # )
        # return sorted(all_names)

    def _full_schema(self, group_key: str, invert: bool) -> Dict[str, pl.DataType]:
        groups = self._groups(group_key, invert)
        result: Dict[str, pl.DataType] = dict()
        for g in groups:
            result = result | {p.name: p.dtype for p in self._params[g]}

        # result = self._required_schema | {
        #     name: dtype for name, (_default, dtype) in self._optional_schema.items()
        # }
        return result

    def _required_params(self, group_key: str, invert: bool) -> List[Param]:
        # groups = self._groups(group_key)
        # result = set()
        # for g in groups:
        #     result = result | {p for p in self._params[g] if p.default is None}
        result = [p for p in self.params(group_key, invert) if p.default is None]

        return result

    def _optional_params(self, group_key: str, invert: bool) -> List[Param]:
        # groups = self._groups(group_key)
        # result = set()
        # for g in groups:
        #     result = result | {p for p in self._params[g] if p.default is not None}

        result = [p for p in self.params(group_key, invert) if p.default is not None]

        return result

    def _validate(self, group_key: str, df: FrameType, params: FrameType):
        required_param_names = [
            p.name for p in self._required_params(group_key, invert=False)
        ]

        missing_required_params = set(required_param_names).difference(params.columns)
        if len(missing_required_params) > 0:
            raise ValueError(f"Missing required parameters: {missing_required_params}")

        full_schema = self._full_schema(group_key, invert=False)

        mismatch_param_dtype = [
            (name, dtype, full_schema[name])
            for name, dtype in df.schema.items()
            if name in full_schema and dtype != full_schema[name]
        ]

        if len(mismatch_param_dtype) > 0:
            raise ValueError(
                f"Type-mismatch on input frame columns: {mismatch_param_dtype}"
            )

    def _subset_params(self, group_key: str, params: FrameType) -> FrameType:
        params_to_drop = [p.name for p in self.params(group_key, invert=True)]
        result = (
            params.select(pl.exclude(params_to_drop))
            if len(params_to_drop) > 0
            else params
        ).unique(maintain_order=True)
        return result

    def apply(
        self, group_key: str, df: FrameType, params: FrameType
    ) -> Tuple[FrameType, FrameType, List[str]]:
        params_subset = self._subset_params(group_key, params)
        self._validate(group_key, df, params_subset)

        # add optional parameters that weren't supplied in params
        optional_params = self._optional_params(group_key, invert=False)
        # optional_param_names = [p.name for p in self._optional_params(group_key)]
        missing_optional_params = {
            p for p in optional_params if p.name not in params_subset.columns
        }

        if len(missing_optional_params) > 0:
            params_subset = params_subset.with_columns(
                [pl.lit(p.default).alias(p.name) for p in missing_optional_params]
            )

        # attempt to correct dtypes of parameters
        full_schema = self._full_schema(group_key, invert=False)

        params_subset = params_subset.with_columns(
            [cast_dtype(pl.col(name), dtype) for name, dtype in full_schema.items()]
        )

        # if the parameter exists in df and in the params frames, then
        # prefer the df. This is because the params could just be default parameters
        # print a warning?
        # common_values = Grouper.common_values(df, params)
        # non_schema_params = params.drop(*common_values)
        # schema_names = self.names(group_key, invert=True)
        # non_schema_params = params.select(pl.exclude(schema_names))

        common_cats = Grouper.common_categories(df, params_subset)
        if len(common_cats) == 0:
            join_type: JoinStrategy = "cross"

            df_schema_params = set(df.columns).intersection(
                self.names(group_key, invert=False)
            )
            non_schema_params = (
                set(params_subset.columns)
                .difference(self.names(group_key, invert=False))
                .union(df_schema_params)
            )
            if len(params_subset.lazy().collect()) > 1 and len(non_schema_params) == 0:
                raise ValueError(
                    "supplied multiple params without any categories. suggestion: add categories to params."
                )
        else:
            join_type = "left"

        df_with_params = df.join(
            params_subset, on=common_cats, how=join_type
        ).with_columns(
            [
                pl.col(p.name).fill_null(p.default)
                for p in optional_params
                if p.default is not None
            ]
        )

        param_cols = [
            p
            for p in Grouper.columns(params_subset, include_time=False)
            if p not in full_schema.keys()
        ]
        result_cols = (
            pl.Series(values=df.columns + param_cols)
            .unique(maintain_order=True)
            .to_list()
        )

        return df_with_params, params_subset, result_cols

    def __repr__(self) -> str:
        group_parts = []
        for g in self._groups("*", invert=False):
            required = ",".join(
                sorted(
                    [
                        f"{p.name}:{p.dtype}"
                        for p in self._required_params(g, invert=False)
                    ]
                )
            )
            optional = ",".join(
                sorted(
                    [
                        f"{p.name}:{p.dtype}={p.default}"
                        for p in self._optional_params(g, invert=False)
                    ]
                )
            )
            group_parts.append(f"{g}=({required})[{optional}]")

        class_name = type(self).__name__
        group_parts_str = ",".join(group_parts)
        rep = f"{class_name}<{group_parts_str}>"

        return rep

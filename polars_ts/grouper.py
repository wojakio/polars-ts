from typing import List, Iterable, FrozenSet, Tuple
from .types import FrameType

import polars as pl


class Grouper:
    def __init__(self) -> None:
        self._has_defined_spec = False
        self._by: FrozenSet[str] = frozenset()
        self._omitting: FrozenSet[str] = frozenset()
        self._all: bool = False
        self._result_includes_time: bool = False
        self._common: bool = False

    @staticmethod
    def _make_set(*cols) -> Tuple[FrozenSet[str], bool]:
        result = set(cols)
        includes_time = "time" in result
        return frozenset(result.difference(["time"])), includes_time

    @staticmethod
    def by(*cols: str | Iterable[str]) -> "Grouper":
        g = Grouper()
        g._by, g._result_includes_time = Grouper._make_set(*cols)
        g._has_defined_spec = True
        return g

    @staticmethod
    def by_time() -> "Grouper":
        g = Grouper()
        g._result_includes_time = True
        g._has_defined_spec = True
        return g

    @staticmethod
    def by_time_and(*cols: str | Iterable[str]) -> "Grouper":
        g = Grouper()
        g._by, _ = Grouper._make_set(*cols)
        g._result_includes_time = True
        g._has_defined_spec = True
        return g

    @staticmethod
    def omitting(*cols: str | Iterable[str]) -> "Grouper":
        g = Grouper()
        g._omitting, time_omitted = Grouper._make_set(*cols)
        g._result_includes_time = not time_omitted
        g._has_defined_spec = True
        return g

    @staticmethod
    def omitting_time_and(*cols: str | Iterable[str]) -> "Grouper":
        g = Grouper()
        g._omitting, _ = Grouper._make_set(*cols)
        g._result_includes_time = False
        g._has_defined_spec = True
        return g

    @staticmethod
    def by_all() -> "Grouper":
        g = Grouper()
        g._all = True
        g._has_defined_spec = True
        return g

    @staticmethod
    def by_time_and_all() -> "Grouper":
        g = Grouper()
        g._all = True
        g._result_includes_time = True
        g._has_defined_spec = True
        return g

    @staticmethod
    def by_common_excluding_time() -> "Grouper":
        g = Grouper()
        g._common = True
        g._has_defined_spec = True
        return g

    @staticmethod
    def by_common_including_time() -> "Grouper":
        g = Grouper()
        g._common = True
        g._result_includes_time = True
        g._has_defined_spec = True
        return g

    @staticmethod
    def common_categories(
        lhs: FrameType, rhs: FrameType, include_time: bool = False
    ) -> List[str]:
        cat_lhs = Grouper.categories(lhs, include_time)
        cat_rhs = Grouper.categories(rhs, include_time)
        cols = set(cat_lhs).intersection(cat_rhs)

        return sorted(cols)

    @staticmethod
    def categories(df: FrameType, include_time: bool) -> List[str]:
        cols = df.select(pl.col(pl.Categorical, pl.Enum)).columns
        if include_time:
            cols = ["time"] + sorted(cols)

        return cols

    @staticmethod
    def values(df: FrameType) -> List[str]:
        cats = set(Grouper.categories(df, include_time=True))
        cols = set(df.columns).difference(cats)
        return sorted(cols)

    @staticmethod
    def numerics(df: FrameType) -> List[str]:
        cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        return sorted(cols)

    def apply(self, *dfs: FrameType) -> List[str]:
        if self._common and len(dfs) < 2:
            raise ValueError("Require at least 2 dataframes when 'common' flag is set.")

        if not self._common and len(dfs) != 1:
            raise ValueError("Too many arguments passed to apply.")

        if self._common:
            df = dfs[0]
            df_other = dfs[1]
            df_has_time = ("time" in df) and ("time" in df_other)
            cols = set(Grouper.common_categories(df, df_other))
        else:
            if not self._has_defined_spec:
                self._all = True

            df = dfs[0]
            df_has_time = "time" in df
            cols = set(Grouper.categories(df, include_time=False))

            if len(self._by) > 0:
                cols = cols.intersection(self._by)
            elif len(self._omitting) > 0:
                cols = cols.difference(self._omitting)
            elif self._all:
                pass
            elif len(self._by) == 0 and len(self._omitting) == 0:
                cols = set()

        if self._result_includes_time and df_has_time:
            result = ["time"] + sorted(cols)
        else:
            result = sorted(cols)

        if len(result) == 0:
            raise ValueError("Empty Grouper Specification")

        return result

    def __repr__(self) -> str:
        if len(self._omitting) > 0:
            time_clause = "TimeAnd" if self._result_includes_time else ""
            return f"Omitting{time_clause}({','.join(sorted(self._omitting))})"

        if len(self._by) > 0:
            time_clause = "TimeAnd" if self._result_includes_time else ""
            return f"By{time_clause}({','.join(sorted(self._by))})"

        if self._all:
            time_clause = "TimeAnd" if self._result_includes_time else ""
            return f"By{time_clause}All"

        return "ByNone"

from typing import List, Iterable, FrozenSet, Tuple
from .types import FrameType

import polars as pl


class Grouper:
    def __init__(self) -> None:
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
        return g

    @staticmethod
    def by_time() -> "Grouper":
        g = Grouper()
        g._result_includes_time = True
        return g

    @staticmethod
    def by_time_and(*cols: str | Iterable[str]) -> "Grouper":
        g = Grouper()
        g._by, _ = Grouper._make_set(*cols)
        g._result_includes_time = True
        return g

    @staticmethod
    def omitting(*cols: str | Iterable[str]) -> "Grouper":
        g = Grouper()
        g._omitting, time_omitted = Grouper._make_set(*cols)
        g._result_includes_time = not time_omitted
        return g

    @staticmethod
    def omitting_time_and(*cols: str | Iterable[str]) -> "Grouper":
        g = Grouper()
        g._omitting, _ = Grouper._make_set(*cols)
        g._result_includes_time = False
        return g

    @staticmethod
    def by_all() -> "Grouper":
        g = Grouper()
        g._all = True
        return g

    @staticmethod
    def by_time_and_all() -> "Grouper":
        g = Grouper()
        g._all = True
        g._result_includes_time = True
        return g

    @staticmethod
    def by_common() -> "Grouper":
        g = Grouper()
        g._common = True
        return g

    @staticmethod
    def by_time_and_common() -> "Grouper":
        g = Grouper()
        g._common = True
        g._result_includes_time = True
        return g

    @staticmethod
    def _get_common_categories_without_time(
        lhs: FrameType, rhs: FrameType
    ) -> List[str]:
        cat_lhs = Grouper._get_categories_without_time(lhs)
        cat_rhs = Grouper._get_categories_without_time(rhs)
        common = set(cat_lhs).intersection(cat_rhs)

        return sorted(common)

    @staticmethod
    def _get_categories_without_time(df: FrameType) -> List[str]:
        return df.select(pl.col(pl.Categorical, pl.Enum)).columns

    def apply(self, *dfs: FrameType) -> List[str]:
        if self._common and len(dfs) < 2:
            raise ValueError("Require at least 2 dataframes when 'common' flag is set.")

        if not self._common and len(dfs) != 1:
            raise ValueError("Too many arguments passed to apply.")

        if self._common:
            df = dfs[0]
            df_other = dfs[1]
            df_has_time = ("time" in df) and ("time" in df_other)
            cols = set(Grouper._get_common_categories_without_time(df, df_other))
        else:
            df = dfs[0]
            df_has_time = "time" in df
            cols = set(Grouper._get_categories_without_time(df))

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

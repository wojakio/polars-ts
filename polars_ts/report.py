from typing import Generic, Optional
import polars as pl

import plotly.graph_objects as go

from .sf import SeriesFrame
from .types import FrameType

__NAMESPACE = "report"


@pl.api.register_lazyframe_namespace(__NAMESPACE)
class ReportFrame(SeriesFrame, Generic[FrameType]):
    def __init__(self, df: FrameType) -> None:
        super().__init__(df)

    def plot(self, title: Optional[str] = None) -> go.Figure:
        fig = go.Figure()

        df = self._df.lazy().collect()

        x_data = df["time"]
        x_range = [x_data.min(), x_data.max()]
        initial_x_range = [
            df.with_columns(pl.max("time").dt.offset_by("-1y")).item(0, "time"),
            x_range[1],
        ]

        layout = {
            "xaxis": dict(
                range=initial_x_range,
                rangeslider=dict(
                    autorange=True,
                ),
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=1, label="1m", step="month", stepmode="backward"
                            ),
                            dict(
                                count=6, label="6m", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="ytd", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                type="date",
            ),
            "title": title,
            "dragmode": "zoom",
            "hovermode": "x",
            "legend": dict(traceorder="reversed"),
            "height": 600,
            "template": "plotly_white",
            "margin": dict(t=100, b=100),
        }

        data_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
        n_plots = len(data_cols)

        for i, col in enumerate(data_cols):
            data_range = [df[col].min(), df[col].max()]
            fig.add_trace(go.Scatter(x=x_data, y=df[col], name=col, yaxis=f"y{i+1}"))

            layout.update(
                {
                    f"yaxis{'' if i == 0 else i+1}": dict(
                        anchor="x",
                        autorange=True,
                        domain=[i / n_plots, (i + 1) / n_plots],
                        # linecolor="#E91E63",
                        mirror=True,
                        range=data_range,
                        showline=True,
                        side="left",
                        # tickfont={"color": "#E91E63"},
                        title=col,
                        tickmode="auto",
                        ticks="",
                        # titlefont={"color": "#E91E63"},
                        type="linear",
                        zeroline=False,
                    )
                }
            )

        fig.update_layout(layout)
        fig.update_traces(
            hoverinfo="x+name+y",
            line={"width": 0.5},
            marker={"size": 0},
            mode="lines",
            showlegend=False,
        )

        return fig

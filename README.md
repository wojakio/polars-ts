# polars-ts

## Plan

- everything is a lazyframe

## Namespaces

### .sf
SeriesFrame structure
- category columns
- non-category = value columns

### .tsf
TimeSeriesFrame structure
- extends SeriesFrame with a `time` column

### .ops
```
- diff/lag?
```

### .time
Manipulation of the time column
```
- resampling
- alignment of: dataframes, time-axis, ...
- business rolling/holiday calendars
- start-points/end-points
```

### .dummy
Relating to generation of dummy data
```
- random walk
- correlated feeds
```

### .math
Core Mathematical Functions

All functions multi-width

```
- covariance
- cum_(prod|sum|min|max)
- ewma
- limit_change
- rle
- rolling (mean|median|quantile|rank|sd|variance|sum|skew|min|max|corr)
```

### .futures

```
- imm conversion
- roll calendar
- continuous contracts
```

### .convert
Conversion functions

the dataframe has either:
- a `<value>_unit` column
- a `unit` column (fallback)

builds closures

```
- fx matrix
- general: km -> miles, kwh -> btu, etc
```

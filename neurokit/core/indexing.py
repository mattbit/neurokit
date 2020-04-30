import pandas as pd
from pandas.core.dtypes.common import (_TD_DTYPE)


class FixedTimedeltaIndex(pd.TimedeltaIndex):
    _supports_partial_string_indexing = False
    _typ = "fixedtimedeltaindex"

    def __new__(
            cls,
            data=None,
            unit=None,
            freq=None,
            closed=None,
            dtype=_TD_DTYPE,
            copy=False,
            name=None,
    ):
        if (isinstance(data, pd.TimedeltaIndex)
                and freq is None
                and name is None):
            return cls._simple_new(data._data)

        return super().__new__(cls, data, unit, freq,
                               closed, dtype, copy, name)

    def _maybe_cast_slice_bound(self, label, side, kind):
        """
        If label is a string, cast it to timedelta.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}

        Returns
        -------
        label : object
        """
        if isinstance(label, str):
            return pd.Timedelta(label)

        return super()._maybe_cast_slice_bound(label, side, kind)

    def __add__(self, other):
        return _maybe_cast_to_fixed_timedelta(super().__add__(other))

    def __radd__(self, other):
        return _maybe_cast_to_fixed_timedelta(super().__radd__(other))

    def __sub__(self, other):
        return _maybe_cast_to_fixed_timedelta(super().__sub__(other))

    def __rsub__(self, other):
        return _maybe_cast_to_fixed_timedelta(super().__rsub__(other))


def _maybe_cast_to_fixed_timedelta(obj):
    if (not isinstance(obj, FixedTimedeltaIndex)
            and isinstance(obj, pd.TimedeltaIndex)):
        return FixedTimedeltaIndex(obj)
    return obj


def timedelta_range(*args, **kwargs):
    """Equivalent of `pandas.timedelta_range` for a `FixedTimedeltaIndex`.

    """
    return FixedTimedeltaIndex(pd.timedelta_range(*args, **kwargs))

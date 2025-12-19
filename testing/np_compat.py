"""Small helper to patch NumPy's `array` call for `copy=False` compatibility.

Use `from testing.np_compat import apply_numpy_compat; apply_numpy_compat()` in your notebook
before calling `calculate_builtin_properties()` to avoid the NumPy 2.0 `copy=False` error.

This is a minimal, local workaround for the runtime only; prefer patching the offending
library or installing NumPy 1.x for a permanent fix.
"""

from typing import Any
import numpy as _np

_orig_array = _np.array


def apply_numpy_compat():
    """Monkeypatch numpy.array to treat `copy=False` like `np.asarray`.

    This makes code that calls `np.array(obj, copy=False)` behave like `np.asarray(obj)`
    which avoids the NumPy 2.0 ValueError for heterogeneous inputs.
    """
    def _compat_array(obj: Any, *args, **kwargs):
        if 'copy' in kwargs and kwargs.get('copy') is False:
            kwargs.pop('copy', None)
            return _np.asarray(obj, dtype=object, **{k: v for k, v in kwargs.items() if k != 'dtype'})
        # handle positional copy argument (rare): if third positional arg is False
        if len(args) >= 1 and args[-1] is False:
            # If user passed copy as positional arg and set it to False, return object-dtype array
            new_args = args[:-1]
            return _np.asarray(obj, dtype=object, **{k: v for k, v in kwargs.items() if k != 'dtype'})
        return _orig_array(obj, *args, **kwargs)

    _np.array = _compat_array


if __name__ == '__main__':
    print('Apply with:')
    print('from testing.np_compat import apply_numpy_compat; apply_numpy_compat()')

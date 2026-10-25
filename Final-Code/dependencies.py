"""Check optional runtime dependencies without embedding any input locations.

Core: numpy, pandas, scipy, torch. Calibration/Ridge: scikit-learn.
Tree benchmark: lightgbm. Verification: pytest.
Run this module directly to list missing imports.
"""
import importlib.util


def missing_dependencies():
    return [name for name in ('numpy', 'pandas', 'scipy', 'torch', 'sklearn', 'lightgbm', 'pytest')
            if importlib.util.find_spec(name) is None]


if __name__ == '__main__':
    print('\n'.join(missing_dependencies()) or 'All runtime dependencies are available.')

"""CSV source of truth, disposable cache, and polar interpolation lookups."""
from pathlib import Path
import os
import pickle
import tempfile
from threading import RLock

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import QhullError
from core.paths import DEFAULT_DATABASE, LEGACY_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
COLUMNS = ['airfoil', 'thickness_pct', 'max_thickness_pos_pct', 'le_radius_ratio',
           'max_camber_pct', 'reynolds', 'alpha', 'cl', 'cd', 'drag_counts',
           'ld', 'cm', 'top_xtr', 'bot_xtr', 'converged']
COLUMNS.insert(1, 'generator_method')
NUMERIC_COLUMNS = [c for c in COLUMNS if c not in ('airfoil', 'generator_method', 'converged')]


class _LazyInterpolators(dict):
    """Preserve legacy mapping access while constructing only queried curves."""
    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def __missing__(self, key):
        if isinstance(key, tuple) and len(key) == 3:
            return self.manager.get_polar_interpolator(*key)
        if isinstance(key, tuple) and len(key) == 2:
            return self.manager.get_polar_interpolator(*key)
        return self.manager.get_polar_interpolator(key)


class Database3DManager:
    def __init__(self, csv_path=None, cache_path=None):
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_DATABASE
        self.cache_path = Path(cache_path) if cache_path else self.csv_path.with_name(self.csv_path.stem + '_Cache.pkl')
        self.df = pd.DataFrame(columns=COLUMNS)
        self.interpolators = _LazyInterpolators(self)
        self.available_airfoils = set()
        self.available_airfoils_by_method = {'warp': set(), 'cst': set()}
        self._lock = RLock()

    @staticmethod
    def _normalize(frame):
        frame = frame.copy()
        if 'le_radius_ratio' not in frame:
            frame['le_radius_ratio'] = 1.0
        if 'generator_method' not in frame:
            frame['generator_method'] = 'warp'
        if not frame.generator_method.isin(['warp', 'cst']).all():
            raise ValueError('generator_method must be warp or cst')
        missing = set(COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing database columns: {sorted(missing)}')
        frame = frame[COLUMNS]
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors='raise')
        converged = frame.converged.astype(str).str.lower().isin(['true', '1', '1.0'])
        finite = np.isfinite(frame[NUMERIC_COLUMNS].to_numpy(dtype=float)).all(axis=1)
        return frame.loc[converged & finite & (frame.cd > 0)].assign(converged=True).reset_index(drop=True)

    def _signature(self):
        stat = self.csv_path.stat()
        return (str(self.csv_path.resolve()), stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _atomic_write(path, writer):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
        os.close(fd)
        try:
            writer(temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _save_cache(self):
        payload = {'version': 1, 'signature': self._signature(), 'df': self.df}
        self._atomic_write(self.cache_path, lambda p: pd.to_pickle(payload, p))

    def _rebuild(self):
        self.interpolators = _LazyInterpolators(self)
        counts = self.df.groupby(['generator_method', 'airfoil']).alpha.nunique()
        self.available_airfoils_by_method = {
            method: {name for (kind, name), count in counts.items() if kind == method and count >= 5}
            for method in ('warp', 'cst')}
        self.available_airfoils = self.available_airfoils_by_method['warp'].copy()

    def get_polar_interpolator(self, airfoil_name, reynolds=None, generator_method='warp'):
        """Lazily cache alpha→Cd or (Re, alpha)→Cd; unsupported keys raise KeyError.

        A supplied Reynolds number selects an exact stored polar, consistent
        with the previous (name, Re) mapping API. No angular extrapolation.
        """
        if generator_method not in ('warp', 'cst'):
            raise ValueError('generator_method must be warp or cst')
        key = airfoil_name if reynolds is None else (airfoil_name, float(reynolds))
        if generator_method != 'warp':
            key = (airfoil_name, None if reynolds is None else float(reynolds), generator_method)
        with self._lock:
            if key in self.interpolators:
                return dict.__getitem__(self.interpolators, key)
            group = self.df.loc[(self.df.airfoil == airfoil_name) &
                                (self.df.generator_method == generator_method)]
            if reynolds is not None:
                polar = group.loc[group.reynolds == reynolds]
                polar = polar.groupby('alpha', as_index=False).cd.mean().sort_values('alpha')
                if len(polar) < 2:
                    raise KeyError(key)
                result = interp1d(polar.alpha, polar.cd, bounds_error=False, fill_value=np.nan)
                self.interpolators[key] = result
                return result
            points = group.groupby(['reynolds', 'alpha'], as_index=False).cd.mean()
            if points.reynolds.nunique() > 1 and points.alpha.nunique() > 1:
                try:
                    result = LinearNDInterpolator(
                        points[['reynolds', 'alpha']].to_numpy(), points.cd.to_numpy(), rescale=True)
                except QhullError:
                    raise KeyError(key) from None
                self.interpolators[key] = result
                return result
            raise KeyError(key)

    def load_database(self, legacy_csv_path=None):
        with self._lock:
            self.df = pd.DataFrame(columns=COLUMNS)
            cached = False
            if self.csv_path.exists():
                if self.cache_path.exists():
                    try:
                        payload = pd.read_pickle(self.cache_path)
                        if payload['version'] == 1 and payload['signature'] == self._signature():
                            self.df = self._normalize(payload['df'])
                            cached = True
                    except (OSError, ValueError, TypeError, KeyError, EOFError, AttributeError, pickle.UnpicklingError):
                        pass
                if not cached:
                    self.df = self._normalize(pd.read_csv(self.csv_path))
            legacy = Path(legacy_csv_path) if legacy_csv_path else LEGACY_DIR / 'Fairing_Airfoil_Database.csv'
            imported = False
            if legacy.is_file() and legacy.resolve() != self.csv_path.resolve():
                existing = set(self.df.loc[(self.df.le_radius_ratio == 1.0) &
                                            (self.df.generator_method == 'warp'), 'airfoil'])
                old = pd.read_csv(legacy)
                old['le_radius_ratio'] = 1.0
                old['generator_method'] = 'warp'
                old = self._normalize(old)
                old = old.loc[~old.airfoil.isin(existing)]
                if not old.empty:
                    self.df = pd.concat([self.df, old], ignore_index=True) if not self.df.empty else old
                    imported = True
            if imported:
                self._atomic_write(self.csv_path, lambda p: self.df.to_csv(p, index=False))
            if self.csv_path.exists() and (imported or not cached):
                self._save_cache()
            self._rebuild()
            return not self.df.empty

    def append_records(self, new_records):
        if not new_records:
            return 0
        new = self._normalize(pd.DataFrame(new_records))
        if new.empty:
            return 0
        with self._lock:
            # Re-read disk to preserve existing records when append is the first call.
            prior = self._normalize(pd.read_csv(self.csv_path)) if self.csv_path.exists() else self.df
            combined = pd.concat([prior, new], ignore_index=True) if not prior.empty else new
            self._atomic_write(self.csv_path, lambda p: combined.to_csv(p, index=False))
            self.df = combined
            self._save_cache()
            self._rebuild()
        return len(new)

    def get_completed_keys(self, generator_method='warp'):
        if generator_method not in ('warp', 'cst'):
            raise ValueError('generator_method must be warp or cst')
        if self.df.empty:
            return set()
        columns = ['thickness_pct', 'max_thickness_pos_pct', 'le_radius_ratio']
        counts = self.df.loc[self.df.generator_method == generator_method].groupby(columns).alpha.nunique()
        return {(round(tc, 1), round(xt, 1), round(kle, 2))
                for (tc, xt, kle), count in counts.items() if count >= 5}

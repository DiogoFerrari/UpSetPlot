from __future__ import print_function, division, absolute_import
from numbers import Number

import pandas as pd
import numpy as np


def generate_data(seed=0, n_samples=10000, n_sets=3, aggregated=False):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({'value': np.zeros(n_samples)})
    for i in range(n_sets):
        r = rng.rand(n_samples)
        df['set%d' % i] = r > rng.rand()
        df['value'] += r

    df.set_index(['set%d' % i for i in range(n_sets)], inplace=True)
    if aggregated:
        return df.value.groupby(level=list(range(n_sets))).count()
    return df.value


def from_memberships(memberships, data=None):
    """Load data where each sample has a collection of set names

    The output should be suitable for passing to `UpSet` or `plot`.

    Parameters
    ----------
    memberships : sequence of collections of strings
        Each element corresponds to a data point, indicating the sets it is a
        member of.  Each set is named by a string.
    data : Series-like or DataFrame-like, optional
        If given, the index of set memberships is attached to this data.
        It must have the same length as `memberships`.
        If not given, the series will contain the value 1.

    Returns
    -------
    DataFrame or Series
        `data` is returned with its index indicating set membership.
        It will be a Series if `data` is a Series or 1d numeric array.
        The index will have levels ordered by set names.

    Examples
    --------
    >>> from upsetplot import from_memberships
    >>> from_memberships([
    ...     ['set1', 'set3'],
    ...     ['set2', 'set3'],
    ...     ['set1'],
    ...     []
    ... ])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    set1   set2   set3
    True   False  True     1
    False  True   True     1
    True   False  False    1
    False  False  False    1
    Name: ones, dtype: ...
    >>> # now with data:
    >>> import numpy as np
    >>> from_memberships([
    ...     ['set1', 'set3'],
    ...     ['set2', 'set3'],
    ...     ['set1'],
    ...     []
    ... ], data=np.arange(12).reshape(4, 3))  # doctest: +NORMALIZE_WHITESPACE
                       0   1   2
    set1  set2  set3
    True  False True   0   1   2
    False True  True   3   4   5
    True  False False  6   7   8
    False False False  9  10  11
    """
    df = pd.DataFrame([{name: True for name in names}
                       for names in memberships])
    for set_name in df.columns:
        if not hasattr(set_name, 'lower'):
            raise ValueError('Set names should be strings')
    if df.shape[1] == 0:
        raise ValueError('Require at least one set. None were found.')
    df.sort_index(axis=1, inplace=True)
    df.fillna(False, inplace=True)
    df = df.astype(bool)
    df.set_index(list(df.columns), inplace=True)
    if data is None:
        return df.assign(ones=1)['ones']

    if hasattr(data, 'loc'):
        data = data.copy(deep=False)
    elif len(data) and isinstance(data[0], Number):
        data = pd.Series(data)
    else:
        data = pd.DataFrame(data)
    if len(data) != len(df):
        raise ValueError('memberships and data must have the same length. '
                         'Got len(memberships) == %d, len(data) == %d'
                         % (len(memberships), len(data)))
    data.index = df.index
    return data


def from_contents(contents, data=None):
    """Build data from category listings

    Parameters
    ----------
    contents : Mapping of strings to sets
        Map values be sets of identifiers (int or string).
    data : DataFrame, optional
        If provided, this should be indexed by the identifiers used in
        `contents`.

    Returns
    -------
    DataFrame
    """
    cat_series = [pd.Series(True, index=list(elements), name=name)
                  for name, elements in contents.items()]
    df = pd.concat(cat_series, axis=1, sort=False)
    df.fillna(False, inplace=True)
    set_names = list(df.columns)
    if data:
        if set(df.columns).intersection(data.columns):
            raise ValueError('Data columns overlap with category naems')

        df = pd.concat([df, data], axis=1, sort=False)
    return df.reset_index().set_index(set_names)


### SPEC


# Could also be "CategorizedData"
class VennData:
    def __init__(self, df, key_fields=None, category_fields=None):
        self._df = self._check_df(df)

    def _check_df(self, df):
        # TODO
        return df

    @classmethod
    def from_memberships(cls, memberships, data=None):
        """Build data from the category membership of each element

        Parameters
        ----------
        memberships : sequence of collections of strings
            Each element corresponds to a data point, indicating the sets it is
            a member of.  Each set is named by a string.
        data : Series-like or DataFrame-like, optional
            If given, the index of set memberships is attached to this data.
            It must have the same length as `memberships`.
            If not given, the series will contain the value 1.

        Returns
        -------
        VennData
        """
        return cls(from_memberships(memberships, data))

    @classmethod
    def from_contents(cls, contents, data=None):
        """Build data from category listings

        Parameters
        ----------
        contents : Mapping of strings to sets
            Map values be sets of identifiers (int or string).
        data : DataFrame, optional
            If provided, this should be indexed by the identifiers used in
            `contents`.

        Returns
        -------
        VennData
        """
        return cls(from_contents(contents, data))

    def _get_cat_mask(self):
        return self._df.index.to_frame(index=False)

    def _get_data(self):
        return self._df.reset_index()

    def get_intersection(self, categories, inclusive=False):
        """Retrieve elements that are in all the given categories

        Parameters
        ----------
        categories : collection of strings
        inclusive : bool
            If False (default), do not include elements that are in additional
            categories.
        """
        categories = list(categories)
        cat_mask = self._get_cat_mask()
        # XXX: More efficient with a groupby?
        mask = cat_mask[categories].all(axis=1)
        if not inclusive:
            mask &= ~cat_mask.drop(categories, axis=1).any(axis=1)
        return self._get_data()[mask]

    def count_intersection(self, categories, inclusive=False):
        """Count the number of elements in all the given categories

        Parameters
        ----------
        categories : collection of strings
        inclusive : bool
            If False (default), do not include elements that are in additional
            categories.
        """

from __future__ import print_function, division, absolute_import

try:
    import typing
except ImportError:
    import collections as typing

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import patches
from matplotlib.tight_layout import get_renderer
from numbers import Number
import functools
import distutils
import warnings

def generate_samples(seed=0, n_samples=10000, n_categories=3):
    """Generate artificial samples assigned to set intersections

    Parameters
    ----------
    seed : int
        A seed for randomisation
    n_samples : int
        Number of samples to generate
    n_categories : int
        Number of categories (named "cat0", "cat1", ...) to generate

    Returns
    -------
    DataFrame
        Field 'value' is a weight or score for each element.
        Field 'index' is a unique id for each element.
        Index includes a boolean indicator mask for each category.

        Note: Further fields may be added in future versions.

    See Also
    --------
    generate_counts : Generates the counts for each subset of categories
        corresponding to these samples.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({'value': np.zeros(n_samples)})
    for i in range(n_categories):
        r = rng.rand(n_samples)
        df['cat%d' % i] = r > rng.rand()
        df['value'] += r

    df.reset_index(inplace=True)
    df.set_index(['cat%d' % i for i in range(n_categories)], inplace=True)
    return df


def generate_counts(seed=0, n_samples=10000, n_categories=3):
    """Generate artificial counts corresponding to set intersections

    Parameters
    ----------
    seed : int
        A seed for randomisation
    n_samples : int
        Number of samples to generate statistics over
    n_categories : int
        Number of categories (named "cat0", "cat1", ...) to generate

    Returns
    -------
    Series
        Counts indexed by boolean indicator mask for each category.

    See Also
    --------
    generate_samples : Generates a DataFrame of samples that these counts are
        derived from.
    """
    df = generate_samples(seed=seed, n_samples=n_samples,
                          n_categories=n_categories)
    return df.value.groupby(level=list(range(n_categories))).count()


def generate_data(seed=0, n_samples=10000, n_sets=3, aggregated=False):
    warnings.warn('generate_data was replaced by generate_counts in version '
                  '0.3 and will be removed in version 0.4.',
                  DeprecationWarning)
    if aggregated:
        return generate_counts(seed=seed, n_samples=n_samples,
                               n_categories=n_sets)
    else:
        return generate_samples(seed=seed, n_samples=n_samples,
                                n_categories=n_sets)['value']


def from_indicators(indicators, data=None):
    """Load category membership indicated by a boolean indicator matrix

    This loader also supports the case where the indicator columns can be
    derived from `data`.

    .. versionadded:: 0.6

    Parameters
    ----------
    indicators : DataFrame-like of booleans, Sequence of str, or callable
        Specifies the category indicators (boolean mask arrays) within
        ``data``, i.e. which records in ``data`` belong to which categories.

        If a list of strings, these should be column names found in ``data``
        whose values are boolean mask arrays.

        If a DataFrame, its columns should correspond to categories, and its
        index should be a subset of those in ``data``, values should be True
        where a data record is in that category, and False or NA otherwise.

        If callable, it will be applied to ``data`` after the latter is
        converted to a Series or DataFrame.

    data : Series-like or DataFrame-like, optional
        If given, the index of category membership is attached to this data.
        It must have the same length as `indicators`.
        If not given, the series will contain the value 1.

    Returns
    -------
    DataFrame or Series
        `data` is returned with its index indicating category membership.
        It will be a Series if `data` is a Series or 1d numeric array or None.

    Notes
    -----
    Categories with indicators that are all False will be removed.

    Examples
    --------
    >>> import pandas as pd
    >>> from upsetplot import from_indicators

    Just indicators
    >>> indicators = {"cat1": [True, False, True, False],
    ...               "cat2": [False, True, False, False],
    ...               "cat3": [True, True, False, False]}
    >>> from_indicators(indicators)
    cat1   cat2   cat3
    True   False  True     1.0
    False  True   True     1.0
    True   False  False    1.0
    False  False  False    1.0
    Name: ones, dtype: float64

    Where indicators are included within data, specifying columns by name
    >>> data = pd.DataFrame({"value": [5, 4, 6, 4], **indicators})
    >>> from_indicators(["cat1", "cat3"], data=data)
                 value   cat1   cat2   cat3
    cat1  cat3
    True  True       5   True  False   True
    False True       4  False   True   True
    True  False      6   True  False  False
    False False      4  False  False  False

    Making indicators out of all boolean columns
    >>> from_indicators(lambda data: data.select_dtypes(bool), data=data)
                       value   cat1   cat2   cat3
    cat1  cat2  cat3
    True  False True       5   True  False   True
    False True  True       4  False   True   True
    True  False False      6   True  False  False
    False False False      4  False  False  False

    Using a dataset with missing data, we can use missingness as an indicator
    >>> data = pd.DataFrame({"val1": [pd.NA, .7, pd.NA, .9],
    ...                      "val2": ["male", pd.NA, "female", "female"],
    ...                      "val3": [pd.NA, pd.NA, 23000, 78000]})
    >>> from_indicators(pd.isna, data=data)
                       val1    val2   val3
    val1  val2  val3
    True  False True   <NA>    male   <NA>
    False True  True    0.7    <NA>   <NA>
    True  False False  <NA>  female  23000
    False False False   0.9  female  78000
    """
    if data is not None:
        data = _convert_to_pandas(data)

    if callable(indicators):
        if data is None:
            raise ValueError("data must be provided when indicators is "
                             "callable")
        indicators = indicators(data)

    try:
        indicators[0]
    except Exception:
        pass
    else:
        if isinstance(indicators[0], (str, int)):
            if data is None:
                raise ValueError("data must be provided when indicators are "
                                 "specified as a list of columns")
            if isinstance(indicators, tuple):
                raise ValueError("indicators as tuple is not supported")
            # column array
            indicators = data[indicators]

    indicators = pd.DataFrame(indicators).fillna(False).infer_objects()
    # drop all-False (should we be dropping all-True also? making an option?)
    indicators = indicators.loc[:, indicators.any(axis=0)]

    if not all(dtype.kind == 'b' for dtype in indicators.dtypes):
        raise ValueError('The indicators must all be boolean')

    if data is not None:
        if not (isinstance(indicators.index, pd.RangeIndex)
                and indicators.index[0] == 0
                and indicators.index[-1] == len(data) - 1):
            # index is specified on indicators. Need to align it to data
            if not indicators.index.isin(data.index).all():
                raise ValueError("If indicators.index is not the default, "
                                 "all its values must be present in "
                                 "data.index")
            indicators = indicators.reindex(index=data.index, fill_value=False)
    else:
        data = pd.Series(np.ones(len(indicators)), name="ones")

    indicators.set_index(list(indicators.columns), inplace=True)
    data.index = indicators.index

    return data


def _convert_to_pandas(data, copy=True):
    is_series = False
    if hasattr(data, 'loc'):
        if copy:
            data = data.copy(deep=False)
        is_series = data.ndim == 1
    elif len(data):
        try:
            is_series = isinstance(data[0], Number)
        except KeyError:
            is_series = False
    if is_series:
        data = pd.Series(data)
    else:
        data = pd.DataFrame(data)
    return data


def from_memberships(memberships, data=None):
    """Load data where each sample has a collection of category names

    The output should be suitable for passing to `UpSet` or `plot`.

    Parameters
    ----------
    memberships : sequence of collections of strings
        Each element corresponds to a data point, indicating the sets it is a
        member of.  Each category is named by a string.
    data : Series-like or DataFrame-like, optional
        If given, the index of category memberships is attached to this data.
        It must have the same length as `memberships`.
        If not given, the series will contain the value 1.

    Returns
    -------
    DataFrame or Series
        `data` is returned with its index indicating category membership.
        It will be a Series if `data` is a Series or 1d numeric array.
        The index will have levels ordered by category names.

    Examples
    --------
    >>> from upsetplot import from_memberships
    >>> from_memberships([
    ...     ['cat1', 'cat3'],
    ...     ['cat2', 'cat3'],
    ...     ['cat1'],
    ...     []
    ... ])
    cat1   cat2   cat3
    True   False  True     1
    False  True   True     1
    True   False  False    1
    False  False  False    1
    Name: ones, dtype: ...
    >>> # now with data:
    >>> import numpy as np
    >>> from_memberships([
    ...     ['cat1', 'cat3'],
    ...     ['cat2', 'cat3'],
    ...     ['cat1'],
    ...     []
    ... ], data=np.arange(12).reshape(4, 3))
                       0   1   2
    cat1  cat2  cat3
    True  False True   0   1   2
    False True  True   3   4   5
    True  False False  6   7   8
    False False False  9  10  11
    """
    df = pd.DataFrame([{name: True for name in names}
                       for names in memberships])
    for set_name in df.columns:
        if not hasattr(set_name, 'lower'):
            raise ValueError('Category names should be strings')
    if df.shape[1] == 0:
        raise ValueError('Require at least one category. None were found.')
    df.sort_index(axis=1, inplace=True)
    df.fillna(False, inplace=True)
    df = df.astype(bool)
    df.set_index(list(df.columns), inplace=True)
    if data is None:
        return df.assign(ones=1)['ones']

    data = _convert_to_pandas(data)
    if len(data) != len(df):
        raise ValueError('memberships and data must have the same length. '
                         'Got len(memberships) == %d, len(data) == %d'
                         % (len(memberships), len(data)))
    data.index = df.index
    return data


def from_contents(contents, data=None, id_column='id'):
    """Build data from category listings

    Parameters
    ----------
    contents : Mapping (or iterable over pairs) of strings to sets
        Keys are category names, values are sets of identifiers (int or
        string).
    data : DataFrame, optional
        If provided, this should be indexed by the identifiers used in
        `contents`.
    id_column : str, default='id'
        The column name to use for the identifiers in the output.

    Returns
    -------
    DataFrame
        `data` is returned with its index indicating category membership,
        including a column named according to id_column.
        If data is not given, the order of rows is not assured.

    Notes
    -----
    The order of categories in the output DataFrame is determined from
    `contents`, which may have non-deterministic iteration order.

    Examples
    --------
    >>> from upsetplot import from_contents
    >>> contents = {'cat1': ['a', 'b', 'c'],
    ...             'cat2': ['b', 'd'],
    ...             'cat3': ['e']}
    >>> from_contents(contents)
                      id
    cat1  cat2  cat3
    True  False False  a
          True  False  b
          False False  c
    False True  False  d
          False True   e
    >>> import pandas as pd
    >>> contents = {'cat1': [0, 1, 2],
    ...             'cat2': [1, 3],
    ...             'cat3': [4]}
    >>> data = pd.DataFrame({'favourite': ['green', 'red', 'red',
    ...                                    'yellow', 'blue']})
    >>> from_contents(contents, data=data)
                       id favourite
    cat1  cat2  cat3
    True  False False   0     green
          True  False   1       red
          False False   2       red
    False True  False   3    yellow
          False True    4      blue
    """
    cat_series = [pd.Series(True, index=list(elements), name=name)
                  for name, elements in contents.items()]
    if not all(s.index.is_unique for s in cat_series):
        raise ValueError('Got duplicate ids in a category')

    concat = pd.concat
    if distutils.version.LooseVersion(pd.__version__) >= '0.23.0':
        # silence the warning
        concat = functools.partial(concat, sort=False)

    df = concat(cat_series, axis=1)
    if id_column in df.columns:
        raise ValueError('A category cannot be named %r' % id_column)
    df.fillna(False, inplace=True)
    cat_names = list(df.columns)

    if data is not None:
        if set(df.columns).intersection(data.columns):
            raise ValueError('Data columns overlap with category names')
        if id_column in data.columns:
            raise ValueError('data cannot contain a column named %r' %
                             id_column)
        not_in_data = df.drop(data.index, axis=0, errors='ignore')
        if len(not_in_data):
            raise ValueError('Found identifiers in contents that are not in '
                             'data: %r' % not_in_data.index.values)
        df = df.reindex(index=data.index).fillna(False)
        df = concat([data, df], axis=1)
    df.index.name = id_column
    return df.reset_index().set_index(cat_names)



def _aggregate_data(df, subset_size, sum_over):
    """
    Returns
    -------
    df : DataFrame
        full data frame
    aggregated : Series
        aggregates
    """
    _SUBSET_SIZE_VALUES = ['auto', 'count', 'sum']
    if subset_size not in _SUBSET_SIZE_VALUES:
        raise ValueError('subset_size should be one of %s. Got %r'
                         % (_SUBSET_SIZE_VALUES, subset_size))
    if df.ndim == 1:
        # Series
        input_name = df.name
        df = pd.DataFrame({'_value': df})

        if subset_size == 'auto' and not df.index.is_unique:
            raise ValueError('subset_size="auto" cannot be used for a '
                             'Series with non-unique groups.')
        if sum_over is not None:
            raise ValueError('sum_over is not applicable when the input is a '
                             'Series')
        if subset_size == 'count':
            sum_over = False
        else:
            sum_over = '_value'
    else:
        # DataFrame
        if sum_over is False:
            raise ValueError('Unsupported value for sum_over: False')
        elif subset_size == 'auto' and sum_over is None:
            sum_over = False
        elif subset_size == 'count':
            if sum_over is not None:
                raise ValueError('sum_over cannot be set if subset_size=%r' %
                                 subset_size)
            sum_over = False
        elif subset_size == 'sum':
            if sum_over is None:
                raise ValueError('sum_over should be a field name if '
                                 'subset_size="sum" and a DataFrame is '
                                 'provided.')

    gb = df.groupby(level=list(range(df.index.nlevels)), sort=False)
    if sum_over is False:
        aggregated = gb.size()
        aggregated.name = 'size'
    elif hasattr(sum_over, 'lower'):
        aggregated = gb[sum_over].sum()
    else:
        raise ValueError('Unsupported value for sum_over: %r' % sum_over)

    if aggregated.name == '_value':
        aggregated.name = input_name

    return df, aggregated


def _check_index(df):
    # check all indices are boolean
    if not all(set([True, False]) >= set(level)
               for level in df.index.levels):
        raise ValueError('The DataFrame has values in its index that are not '
                         'boolean')
    df = df.copy(deep=False)
    # XXX: this may break if input is not MultiIndex
    kw = {'levels': [x.astype(bool) for x in df.index.levels],
          'names': df.index.names,
          }
    if hasattr(df.index, 'codes'):
        # compat for pandas <= 0.20
        kw['codes'] = df.index.codes
    else:
        kw['labels'] = df.index.labels
    df.index = pd.MultiIndex(**kw)
    return df


def _scalar_to_list(val):
    if not isinstance(val, (typing.Sequence, set)) or isinstance(val, str):
        val = [val]
    return val


def _get_subset_mask(agg, min_subset_size, max_subset_size,
                     min_degree, max_degree,
                     present, absent):
    """Get a mask over subsets based on size, degree or category presence"""
    subset_mask = True
    if min_subset_size is not None:
        subset_mask = np.logical_and(subset_mask, agg >= min_subset_size)
    if max_subset_size is not None:
        subset_mask = np.logical_and(subset_mask, agg <= max_subset_size)
    if (min_degree is not None and min_degree >= 0) or max_degree is not None:
        degree = agg.index.to_frame().sum(axis=1)
        if min_degree is not None:
            subset_mask = np.logical_and(subset_mask, degree >= min_degree)
        if max_degree is not None:
            subset_mask = np.logical_and(subset_mask, degree <= max_degree)
    if present is not None:
        for col in _scalar_to_list(present):
            subset_mask = np.logical_and(
                subset_mask,
                agg.index.get_level_values(col).values)
    if absent is not None:
        for col in _scalar_to_list(absent):
            exclude_mask = np.logical_not(
                agg.index.get_level_values(col).values)
            subset_mask = np.logical_and(subset_mask, exclude_mask)
    return subset_mask


def _filter_subsets(df, agg,
                    min_subset_size, max_subset_size,
                    min_degree, max_degree):
    subset_mask = _get_subset_mask(agg,
                                   min_subset_size=min_subset_size,
                                   max_subset_size=max_subset_size,
                                   min_degree=min_degree,
                                   max_degree=max_degree,
                                   present=None, absent=None)

    if subset_mask is True:
        return df, agg

    agg = agg[subset_mask]
    df = df[df.index.isin(agg.index)]
    return df, agg


def _process_data(df, sort_by, sort_categories_by, subset_size,
                  sum_over, min_subset_size=None, max_subset_size=None,
                  min_degree=None, max_degree=None, reverse=False):
    df, agg = _aggregate_data(df, subset_size, sum_over)
    total = agg.sum()
    df = _check_index(df)
    totals = [agg[agg.index.get_level_values(name).values.astype(bool)].sum()
              for name in agg.index.names]
    totals = pd.Series(totals, index=agg.index.names)

    # filter subsets:
    df, agg = _filter_subsets(df, agg,
                              min_subset_size, max_subset_size,
                              min_degree, max_degree)

    # sort:
    if sort_categories_by == 'cardinality':
        totals.sort_values(ascending=False, inplace=True)
    elif sort_categories_by is not None:
        raise ValueError('Unknown sort_categories_by: %r' % sort_categories_by)
    df = df.reorder_levels(totals.index.values)
    agg = agg.reorder_levels(totals.index.values)

    if sort_by == 'cardinality':
        agg = agg.sort_values(ascending=False)
    elif sort_by == 'degree':
        index_tuples = sorted(agg.index,
                              key=lambda x: (sum(x),) + tuple(reversed(x)))
        agg = agg.reindex(pd.MultiIndex.from_tuples(index_tuples,
                                                    names=agg.index.names))
    elif sort_by is None:
        pass
    else:
        raise ValueError('Unknown sort_by: %r' % sort_by)

    # add '_bin' to df indicating index in agg
    # XXX: ugly!
    def _pack_binary(X):
        X = pd.DataFrame(X)
        out = 0
        for i, (_, col) in enumerate(X.items()):
            out *= 2
            out += col
        return out

    df_packed = _pack_binary(df.index.to_frame())
    data_packed = _pack_binary(agg.index.to_frame())
    df['_bin'] = pd.Series(df_packed).map(
        pd.Series(np.arange(len(data_packed))[::-1 if reverse else 1],
                  index=data_packed))
    if reverse:
        agg = agg[::-1]
    return total, df, agg, totals


def _multiply_alpha(c, mult):
    r, g, b, a = colors.to_rgba(c)
    a *= mult
    return colors.to_hex((r, g, b, a), keep_alpha=True)


class _Transposed:
    """Wrap an object in order to transpose some plotting operations

    Attributes of obj will be mapped.
    Keyword arguments when calling obj will be mapped.

    The mapping is not recursive: callable attributes need to be _Transposed
    again.
    """

    def __init__(self, obj):
        self.__obj = obj

    def __getattr__(self, key):
        return getattr(self.__obj, self._NAME_TRANSPOSE.get(key, key))

    def __call__(self, *args, **kwargs):
        return self.__obj(*args, **{self._NAME_TRANSPOSE.get(k, k): v
                                    for k, v in kwargs.items()})

    _NAME_TRANSPOSE = {
        'width': 'height',
        'height': 'width',
        'hspace': 'wspace',
        'wspace': 'hspace',
        'hlines': 'vlines',
        'vlines': 'hlines',
        'bar': 'barh',
        'barh': 'bar',
        'xaxis': 'yaxis',
        'yaxis': 'xaxis',
        'left': 'bottom',
        'right': 'top',
        'top': 'right',
        'bottom': 'left',
        'sharex': 'sharey',
        'sharey': 'sharex',
        'get_figwidth': 'get_figheight',
        'get_figheight': 'get_figwidth',
        'set_figwidth': 'set_figheight',
        'set_figheight': 'set_figwidth',
        'set_xlabel': 'set_ylabel',
        'set_ylabel': 'set_xlabel',
        'set_xlim': 'set_ylim',
        'set_ylim': 'set_xlim',
        'get_xlim': 'get_ylim',
        'get_ylim': 'get_xlim',
        'set_autoscalex_on': 'set_autoscaley_on',
        'set_autoscaley_on': 'set_autoscalex_on',
    }


def _transpose(obj):
    if isinstance(obj, str):
        return _Transposed._NAME_TRANSPOSE.get(obj, obj)
    return _Transposed(obj)


def _identity(obj):
    return obj


class UpSet:
    """Manage the data and drawing for a basic UpSet plot

    Primary public method is :meth:`plot`.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Elements associated with categories (a DataFrame), or the size of each
        subset of categories (a Series).
        Should have MultiIndex where each level is binary,
        corresponding to category membership.
        If a DataFrame, `sum_over` must be a string or False.
    orientation : {'horizontal' (default), 'vertical'}
        If horizontal, intersections are listed from left to right.
    sort_by : {'cardinality', 'degree', None}
        If 'cardinality', subset are listed from largest to smallest.
        If 'degree', they are listed in order of the number of categories
        intersected. If None, the order they appear in the data input is
        used.

        .. versionchanged:: 0.5
            Setting None was added.
    sort_categories_by : {'cardinality', None}
        Whether to sort the categories by total cardinality, or leave them
        in the provided order.

        .. versionadded:: 0.3
    subset_size : {'auto', 'count', 'sum'}
        Configures how to calculate the size of a subset. Choices are:

        'auto' (default)
            If `data` is a DataFrame, count the number of rows in each group,
            unless `sum_over` is specified.
            If `data` is a Series with at most one row for each group, use
            the value of the Series. If `data` is a Series with more than one
            row per group, raise a ValueError.
        'count'
            Count the number of rows in each group.
        'sum'
            Sum the value of the `data` Series, or the DataFrame field
            specified by `sum_over`.
    sum_over : str or None
        If `subset_size='sum'` or `'auto'`, then the intersection size is the
        sum of the specified field in the `data` DataFrame. If a Series, only
        None is supported and its value is summed.
    min_subset_size : int, optional
        Minimum size of a subset to be shown in the plot. All subsets with
        a size smaller than this threshold will be omitted from plotting.
        Size may be a sum of values, see `subset_size`.

        .. versionadded:: 0.5
    max_subset_size : int, optional
        Maximum size of a subset to be shown in the plot. All subsets with
        a size greater than this threshold will be omitted from plotting.

        .. versionadded:: 0.5
    min_degree : int, optional
        Minimum degree of a subset to be shown in the plot.

        .. versionadded:: 0.5
    max_degree : int, optional
        Maximum degree of a subset to be shown in the plot.

        .. versionadded:: 0.5
    facecolor : 'auto' or matplotlib color or float
        Color for bar charts and active dots. Defaults to black if
        axes.facecolor is a light color, otherwise white.

        .. versionchanged:: 0.6
            Before 0.6, the default was 'black'
    other_dots_color : matplotlib color or float
        Color for shading of inactive dots, or opacity (between 0 and 1)
        applied to facecolor.

        .. versionadded:: 0.6
    shading_color : matplotlib color or float
        Color for shading of odd rows in matrix and totals, or opacity (between
        0 and 1) applied to facecolor.

        .. versionadded:: 0.6
    with_lines : bool
        Whether to show lines joining dots in the matrix, to mark multiple
        categories being intersected.
    element_size : float or None
        Side length in pt. If None, size is estimated to fit figure
    intersection_plot_elements : int
        The intersections plot should be large enough to fit this many matrix
        elements. Set to 0 to disable intersection size bars.

        .. versionchanged:: 0.4
            Setting to 0 is handled.
    totals_plot_elements : int
        The totals plot should be large enough to fit this many matrix
        elements.
    show_counts : bool or str, default=False
        Whether to label the intersection size bars with the cardinality
        of the intersection. When a string, this formats the number.
        For example, '%d' is equivalent to True.
    show_percentages : bool, default=False
        Whether to label the intersection size bars with the percentage
        of the intersection relative to the total dataset.
        This may be applied with or without show_counts.
    label_position : position of the category labels when using 
                     vertical orientation
    tsep : thousands separator
    dec : decimal separator
    digits : number of digits when percentages are shown
    totals_label_position : position of the labels for the total plot when
                            counts are shown
    totals_label_rotation : rotation of the labels
    intersections_label_position : position of the labels for the 
                                   intersection plot when counts are shown
    intersections_label_rotation : rotation of the labels

        .. versionadded:: 0.4
    """
    _default_figsize = (10, 6)

    def __init__(self, data, orientation='horizontal', sort_by='degree',
                 sort_categories_by='cardinality',
                 subset_size='auto', sum_over=None,
                 min_subset_size=None, max_subset_size=None,
                 min_degree=None, max_degree=None,
                 facecolor='auto', other_dots_color=.18, shading_color=.05,
                 with_lines=True, element_size=32,
                 intersection_plot_elements=6, totals_plot_elements=2,
                 show_counts='', show_percentages=False,
                 label_position=None,
                 tsep=',',
                 dec='.',
                 digits=1,
                 totals_label_position=None,
                 totals_label_rotation=None,
                 intersections_label_position=None,
                 intersections_label_rotation=None,
                 scatter_kws=None
                 ):


        self.__tsep__ = ','
        self.__dec__  = '.'
        self.__digits__  = digits
        self._horizontal = orientation == 'horizontal'
        self.__totals_label_position__ = (totals_label_position
                                          if totals_label_position
                                          else ('left' if self._horizontal
                                                else 'top'))
        self.__totals_label_rotation__ = (totals_label_rotation if
                                          totals_label_rotation else 0)
        self.__intersections_label_position__ = (intersections_label_position
                                                 if intersections_label_position
                                                 else ('top'
                                                       if self._horizontal
                                                       else 'right'))
        self.__intersections_label_rotation__ = (intersections_label_rotation
                                                 if intersections_label_rotation
                                                 else 0)
        self._reorient = _identity if self._horizontal else _transpose
        if facecolor == 'auto':
            bgcolor = matplotlib.rcParams.get('axes.facecolor', 'white')
            r, g, b, a = colors.to_rgba(bgcolor)
            lightness = colors.rgb_to_hsv((r, g, b))[-1] * a
            facecolor = 'black' if lightness >= .5 else 'white'
        self._facecolor = facecolor
        self._shading_color = (_multiply_alpha(facecolor, shading_color)
                               if isinstance(shading_color, float)
                               else shading_color)
        self._other_dots_color = (_multiply_alpha(facecolor, other_dots_color)
                                  if isinstance(other_dots_color, float)
                                  else other_dots_color)
        self._with_lines = with_lines
        self._element_size = element_size
        self._totals_plot_elements = totals_plot_elements
        self._subset_plots = [{'type': 'default',
                               'id': 'intersections',
                               'elements': intersection_plot_elements}]
        if not intersection_plot_elements:
            self._subset_plots.pop()
        self._show_counts = show_counts
        self._show_percentages = show_percentages
        self.__scatter_kws__ = scatter_kws
        self.__label_position__ = label_position

        # format data 
        # -----------
        data = data.astype(int).dot(data.columns + ",").str.rstrip(',')
        data = from_memberships(data.str.split(','), data=data)

        (self.total, self._df, self.intersections,
         self.totals) = _process_data(data,
                                      sort_by=sort_by,
                                      sort_categories_by=sort_categories_by,
                                      subset_size=subset_size,
                                      sum_over=sum_over,
                                      min_subset_size=min_subset_size,
                                      max_subset_size=max_subset_size,
                                      min_degree=min_degree,
                                      max_degree=max_degree,
                                      reverse=not self._horizontal)
        self.subset_styles = [{"facecolor": facecolor}
                              for i in range(len(self.intersections))]
        self.subset_legend = []  # pairs of (style, label)

    def _swapaxes(self, x, y):
        if self._horizontal:
            return x, y
        return y, x

    def style_subsets(self, present=None, absent=None,
                      min_subset_size=None, max_subset_size=None,
                      min_degree=None, max_degree=None,
                      facecolor=None, edgecolor=None, hatch=None,
                      linewidth=None, linestyle=None, label=None):
        """Updates the style of selected subsets' bars and matrix dots

        Parameters are either used to select subsets, or to style them with
        attributes of :class:`matplotlib.patches.Patch`, apart from label,
        which adds a legend entry.

        Parameters
        ----------
        present : str or list of str, optional
            Category or categories that must be present in subsets for styling.
        absent : str or list of str, optional
            Category or categories that must not be present in subsets for
            styling.
        min_subset_size : int, optional
            Minimum size of a subset to be styled.
        max_subset_size : int, optional
            Maximum size of a subset to be styled.
        min_degree : int, optional
            Minimum degree of a subset to be styled.
        max_degree : int, optional
            Maximum degree of a subset to be styled.

        facecolor : str or matplotlib color, optional
            Override the default UpSet facecolor for selected subsets.
        edgecolor : str or matplotlib color, optional
            Set the edgecolor for bars, dots, and the line between dots.
        hatch : str, optional
            Set the hatch. This will apply to intersection size bars, but not
            to matrix dots.
        linewidth : int, optional
            Line width in points for edges.
        linestyle : str, optional
            Line style for edges.

        label : str, optional
            If provided, a legend will be added
        """
        style = {"facecolor": facecolor, "edgecolor": edgecolor,
                 "hatch": hatch,
                 "linewidth": linewidth, "linestyle": linestyle}
        style = {k: v for k, v in style.items() if v is not None}
        mask = _get_subset_mask(self.intersections,
                                present=present, absent=absent,
                                min_subset_size=min_subset_size,
                                max_subset_size=max_subset_size,
                                min_degree=min_degree, max_degree=max_degree)
        for idx in np.flatnonzero(mask):
            self.subset_styles[idx].update(style)

        if label is not None:
            if "facecolor" not in style:
                style["facecolor"] = self._facecolor
            for i, (other_style, other_label) in enumerate(self.subset_legend):
                if other_style == style:
                    if other_label != label:
                        self.subset_legend[i] = (style,
                                                 other_label + '; ' + label)
                    break
            else:
                self.subset_legend.append((style, label))

    def _plot_bars(self, ax, data, title, colors=None, use_labels=False):
        ax = self._reorient(ax)
        ax.set_autoscalex_on(False)
        data_df = pd.DataFrame(data)
        if self._horizontal:
            data_df = data_df.loc[:, ::-1]  # reverse: top row is top of stack

        # TODO: colors should be broadcastable to data_df shape
        if callable(colors):
            colors = colors(range(data_df.shape[1]))
        elif isinstance(colors, (str, type(None))):
            colors = [colors] * len(data_df)

        if self._horizontal:
            colors = reversed(colors)

        x = np.arange(len(data_df))
        cum_y = None
        all_rects = []
        for (name, y), color in zip(data_df.items(), colors):
            rects = ax.bar(x, y, .5, cum_y,
                           color=color, zorder=10,
                           label=name if use_labels else None,
                           align='center')
            cum_y = y if cum_y is None else cum_y + y
            all_rects.extend(rects)

        self._label_sizes(ax, rects,
                          self.__intersections_label_position__,
                          self.__intersections_label_rotation__
                          )

        ax.xaxis.set_visible(False)
        ax.grid(b=None, which='major', axis='both', linestyle='-', alpha=.3)
        ax.set_axisbelow(True) # to put the grid below the plot
        for x in ['top', 'bottom', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)

        tick_axis = ax.yaxis
        tick_axis.grid(True)
        ax.set_ylabel(title)
        return all_rects

    def _plot_stacked_bars(self, ax, by, sum_over, colors, title):
        df = self._df.set_index("_bin").set_index(by, append=True, drop=False)
        gb = df.groupby(level=list(range(df.index.nlevels)), sort=True)
        if sum_over is None and "_value" in df.columns:
            data = gb["_value"].sum()
        elif sum_over is None:
            data = gb.size()
        else:
            data = gb[sum_over].sum()
        data = data.unstack(by).fillna(0)
        if isinstance(colors, str):
            colors = matplotlib.cm.get_cmap(colors)
        elif isinstance(colors, typing.Mapping):
            colors = data.columns.map(colors).values
            if pd.isna(colors).any():
                raise KeyError("Some labels mapped by colors: %r" %
                               data.columns[pd.isna(colors)].tolist())

        self._plot_bars(ax, data=data, colors=colors, title=title,
                        use_labels=True)

        handles, labels = ax.get_legend_handles_labels()
        if self._horizontal:
            # Make legend order match visual stack order
            ax.legend(reversed(handles), reversed(labels))
        else:
            ax.legend()

    def add_stacked_bars(self, by, sum_over=None, colors=None, elements=3,
                         title=None):
        """Add a stacked bar chart over subsets when :func:`plot` is called.

        Used to plot categorical variable distributions within each subset.

        .. versionadded:: 0.6

        Parameters
        ----------
        by : str
            Column name within the dataframe for color coding the stacked bars,
            containing discrete or categorical values.
        sum_over : str, optional
            Ordinarily the bars will chart the size of each group. sum_over
            may specify a column which will be summed to determine the size
            of each bar.
        colors : Mapping, list-like, str or callable, optional
            The facecolors to use for bars corresponding to each discrete
            label, specified as one of:

            Mapping
                Maps from label to matplotlib-compatible color specification.
            list-like
                A list of matplotlib colors to apply to labels in order.
            str
                The name of a matplotlib colormap name.
            callable
                When called with the number of labels, this should return a
                list-like of that many colors.  Matplotlib colormaps satisfy
                this callable API.
            None
                Uses the matplotlib default colormap.
        elements : int, default=3
            Size of the axes counted in number of matrix elements.
        title : str, optional
            The axis title labelling bar length.

        Returns
        -------
        None
        """
        # TODO: allow sort_by = {"lexical", "sum_squares", "rev_sum_squares",
        #                        list of labels}
        self._subset_plots.append({'type': 'stacked_bars',
                                   'by': by,
                                   'sum_over': sum_over,
                                   'colors': colors,
                                   'title': title,
                                   'id': 'extra%d' % len(self._subset_plots),
                                   'elements': elements})

    def add_catplot(self, kind, value=None, elements=3, **kw):
        """Add a seaborn catplot over subsets when :func:`plot` is called.

        Parameters
        ----------
        kind : str
            One of {"point", "bar", "strip", "swarm", "box", "violin", "boxen"}
        value : str, optional
            Column name for the value to plot (i.e. y if
            orientation='horizontal'), required if `data` is a DataFrame.
        elements : int, default=3
            Size of the axes counted in number of matrix elements.
        **kw : dict
            Additional keywords to pass to :func:`seaborn.catplot`.

            Our implementation automatically determines 'ax', 'data', 'x', 'y'
            and 'orient', so these are prohibited keys in `kw`.

        Returns
        -------
        None
        """
        assert not set(kw.keys()) & {'ax', 'data', 'x', 'y', 'orient'}
        if value is None:
            if '_value' not in self._df.columns:
                raise ValueError('value cannot be set if data is a Series. '
                                 'Got %r' % value)
        else:
            if value not in self._df.columns:
                raise ValueError('value %r is not a column in data' % value)
        self._subset_plots.append({'type': 'catplot',
                                   'value': value,
                                   'kind': kind,
                                   'id': 'extra%d' % len(self._subset_plots),
                                   'elements': elements,
                                   'kw': kw})

    def _check_value(self, value):
        if value is None and '_value' in self._df.columns:
            value = '_value'
        elif value is None:
            raise ValueError('value can only be None when data is a Series')
        return value

    def _plot_catplot(self, ax, value, kind, kw):
        df = self._df
        value = self._check_value(value)
        kw = kw.copy()
        if self._horizontal:
            kw['orient'] = 'v'
            kw['x'] = '_bin'
            kw['y'] = value
        else:
            kw['orient'] = 'h'
            kw['x'] = value
            kw['y'] = '_bin'
        import seaborn
        kw['ax'] = ax
        getattr(seaborn, kind + 'plot')(data=df, **kw)

        ax = self._reorient(ax)
        if value == '_value':
            ax.set_ylabel('')

        ax.xaxis.set_visible(False)
        for x in ['top', 'bottom', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)

        tick_axis = ax.yaxis
        tick_axis.grid(True)

    def make_grid(self, fig=None):
        """Get a SubplotSpec for each Axes, accounting for label text width
        """
        n_cats = len(self.totals)
        n_inters = len(self.intersections)

        if fig is None:
            fig = plt.gcf()

        # Determine text size to determine figure size / spacing
        r = get_renderer(fig)
        text_kw = {"size": matplotlib.rcParams['xtick.labelsize']}
        # adding "x" ensures a margin
        t = fig.text(0, 0, '\n'.join(str(label) + "x"
                                     for label in self.totals.index.values),
                     **text_kw)
        textw = t.get_window_extent(renderer=r).width
        t.remove()

        figw = self._reorient(fig.get_window_extent(renderer=r)).width

        sizes = np.asarray([p['elements'] for p in self._subset_plots])
        fig = self._reorient(fig)

        non_text_nelems = len(self.intersections) + self._totals_plot_elements
        if self._element_size is None:
            colw = (figw - textw) / non_text_nelems
        else:
            render_ratio = figw / fig.get_figwidth()
            colw = self._element_size / 72 * render_ratio
            figw = colw * (non_text_nelems + np.ceil(textw / colw) + 1)
            fig.set_figwidth(figw / render_ratio)
            fig.set_figheight((colw * (n_cats + sizes.sum())) /
                              render_ratio)

        text_nelems = int(np.ceil(figw / colw - non_text_nelems))
        # print('textw', textw, 'figw', figw, 'colw', colw,
        #       'ncols', figw/colw, 'text_nelems', text_nelems)

        GS = self._reorient(matplotlib.gridspec.GridSpec)
        gridspec = GS(*self._swapaxes(n_cats + (sizes.sum() or 0),
                                      n_inters + text_nelems +
                                      self._totals_plot_elements),
                      hspace=1)
        if self._horizontal:
            out = {'matrix': gridspec[-n_cats:, -n_inters:],
                   'shading': gridspec[-n_cats:, :],
                   'totals': gridspec[-n_cats:, :self._totals_plot_elements],
                   'gs': gridspec}
            cumsizes = np.cumsum(sizes[::-1])
            for start, stop, plot in zip(np.hstack([[0], cumsizes]), cumsizes,
                                         self._subset_plots[::-1]):
                out[plot['id']] = gridspec[start:stop, -n_inters:]
        else:
            out = {'matrix': gridspec[-n_inters:, :n_cats],
                   'shading': gridspec[:, :n_cats],
                   'totals': gridspec[:self._totals_plot_elements, :n_cats],
                   'gs': gridspec}
            cumsizes = np.cumsum(sizes)
            for start, stop, plot in zip(np.hstack([[0], cumsizes]), cumsizes,
                                         self._subset_plots):
                out[plot['id']] = \
                    gridspec[-n_inters:, start + n_cats:stop + n_cats]
        return out

    def plot_matrix(self, ax):
        """Plot the matrix of intersection indicators onto ax
        """
        ax = self._reorient(ax)
        data = self.intersections
        n_cats = data.index.nlevels

        inclusion = data.index.to_frame().values

        # Prepare styling
        styles = [
            [
                self.subset_styles[i]
                if inclusion[i, j]
                else {"facecolor": self._other_dots_color, "linewidth": 0}
                for j in range(n_cats)
            ]
            for i in range(len(data))
        ]
        styles = sum(styles, [])  # flatten nested list
        style_columns = {"facecolor": "facecolors",
                         "edgecolor": "edgecolors",
                         "linewidth": "linewidths",
                         "linestyle": "linestyles",
                         "hatch": "hatch"}
        styles = pd.DataFrame(styles).reindex(columns=style_columns.keys())
        styles["linewidth"].fillna(1, inplace=True)
        styles["facecolor"].fillna(self._facecolor, inplace=True)
        styles["edgecolor"].fillna(styles["facecolor"], inplace=True)
        styles["linestyle"].fillna("solid", inplace=True)
        del styles["hatch"]  # not supported in matrix (currently)

        x = np.repeat(np.arange(len(data)), n_cats)
        y = np.tile(np.arange(n_cats), len(data))

        # Plot dots
        # if self._element_size is not None:
        #     s = (self._element_size * .35) ** 2
        # else:
        #     # TODO: make s relative to colw
        #     s = 200
        ax.scatter(*self._swapaxes(x, y), 
                   zorder=10,
                   **styles.rename(columns=style_columns),
                   **self.__scatter_kws__,
                   )

        # Plot lines
        if self._with_lines:
            idx = np.flatnonzero(inclusion)
            line_data = (pd.Series(y[idx], index=x[idx])
                         .groupby(level=0)
                         .aggregate(['min', 'max']))
            colors = pd.Series([
                style.get("edgecolor", style.get("facecolor", self._facecolor))
                for style in self.subset_styles],
                name="color")
            line_data = line_data.join(colors)
            ax.vlines(line_data.index.values,
                      line_data['min'], line_data['max'],
                      lw=2, colors=line_data["color"],
                      zorder=5)

        # Ticks and axes
        if self._horizontal:
            ha = 'right' 
            rotation = 0
        if not self._horizontal and not self.__label_position__:
            ha = 'right' 
            rotation = 30
        tick_axis = ax.yaxis
        tick_axis.set_ticks(np.arange(n_cats))
        tick_axis.set_ticklabels(data.index.names,
                                 rotation=rotation,
                                 ha=ha)
        ax.xaxis.set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)
        if not self._horizontal:
            if not self.__label_position__:
                ax.yaxis.set_ticks_position('bottom')
            else:
                ax.yaxis.set_ticks_position(self.__label_position__)
        ax.set_frame_on(False)
        ax.set_xlim(-.5, x[-1] + .5, auto=False)
        ax.grid(False)

    def plot_intersections(self, ax):
        """Plot bars indicating intersection size
        """
        rects = self._plot_bars(ax, self.intersections,
                                title='Intersection size',
                                colors=self._facecolor)
        for style, rect in zip(self.subset_styles, rects):
            style = style.copy()
            style.setdefault("edgecolor",
                             style.get("facecolor", self._facecolor))
            for attr, val in style.items():
                getattr(rect, "set_" + attr)(val)

        if self.subset_legend:
            styles, labels = zip(*self.subset_legend)
            styles = [patches.Patch(**patch_style) for patch_style in styles]
            ax.legend(styles, labels)

    def _label_sizes(self, ax, rects, where, rotation=0):
        if not self._show_counts and not self._show_percentages:
            return

        def format_text(val):
            if self._show_counts and not self._show_percentages:
                return f"{int(val):,}".replace(",", self.__tsep__)
            if self._show_counts and self._show_percentages:
                counts = f"{int(val):,}".replace(",", self.__tsep__)
                perc = (f"{100 * val / self.total:.{self.__digits__}f}"
                        .replace('.', self.__dec__))
                if where == 'top' and rotation ==0:
                    return f"{counts}\n({perc}%)"
                if where == 'top' and rotation ==0:
                    return f"{counts}\n({perc}%)"
                if where == 'top' and rotation !=0:
                    return f"{counts} ({perc}%)"
                if where == 'left':
                    return f"  {counts} ({perc}%)"
        # if self._show_counts is True:
        #     count_fmt = "%d"
        # else:
        #     count_fmt = self._show_counts

        # if self._show_percentages is True:
        #     pct_fmt = "%.1f%%"
        # else:
        #     pct_fmt = self._show_percentages

        # if count_fmt and pct_fmt:
        #     if where == 'top' and rotation ==0:
        #         fmt = '%s\n(%s)' % (count_fmt, pct_fmt)
        #     elif where == 'top' and rotation !=0:
        #         fmt = '%s (%s)' % (count_fmt, pct_fmt)
        #     elif where == 'left':
        #         fmt = '  %s (%s)' % (count_fmt, pct_fmt)

        #     def make_args(val):
        #         return val, 100 * val / self.total
        # elif count_fmt:
        #     fmt = count_fmt

        #     def make_args(val):
        #         return val,
        # else:
        #     fmt = pct_fmt

        #     def make_args(val):
        #         return 100 * val / self.total,

            

        if where == 'right':
            margin = 0.01 * abs(np.diff(ax.get_xlim()))
            for rect in rects:
                width = rect.get_width() + rect.get_x()
                ax.text(width + margin,
                        rect.get_y() + rect.get_height() * .5,
                        f"{format_text(width)}",
                        # fmt % make_args(width),
                        ha='left', va='center',
                        rotation=rotation
                        )
        elif where == 'left':
            margin = 0.01 * abs(np.diff(ax.get_xlim()))
            for rect in rects:
                width = rect.get_width() + rect.get_x()
                ax.text(width + margin,
                        rect.get_y() + rect.get_height() * .5,
                        f"{format_text(width)}",
                        # fmt % make_args(width),
                        ha='right', va='center',
                        rotation=rotation
                        )
        elif where == 'top':
            margin = 0.01 * abs(np.diff(ax.get_ylim()))
            for rect in rects:
                height = rect.get_height() + rect.get_y()
                ax.text(rect.get_x() + rect.get_width() * .5,
                        height + margin,
                        # fmt % make_args(height),
                        f"{format_text(height)}",
                        ha='center', va='bottom',
                        rotation=rotation
                        )
        else:
            raise NotImplementedError('unhandled where: %r' % where)

    def plot_totals(self, ax):
        """Plot bars indicating total set size
        """
        orig_ax = ax
        ax = self._reorient(ax)
        rects = ax.barh(np.arange(len(self.totals.index.values)), self.totals,
                        .5, color=self._facecolor, align='center')
        self._label_sizes(ax, rects,
                          self.__totals_label_position__,
                          self.__totals_label_rotation__)

        max_total = self.totals.max()
        if self._horizontal:
            orig_ax.set_xlim(max_total, 0)
        for x in ['top', 'left', 'right']:
            ax.spines[self._reorient(x)].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        # 
        ax.patch.set_visible(False)
        ax.grid(b=None, which='major', axis='both', linestyle='-', alpha=.3)
        ax.set_axisbelow(True) # to put the grid below the plot

    def plot_shading(self, ax):
        # alternating row shading (XXX: use add_patch(Rectangle)?)
        for i in range(0, len(self.totals), 2):
            rect = plt.Rectangle(self._swapaxes(0, i - .4),
                                 *self._swapaxes(*(1, .8)),
                                 facecolor=self._shading_color, lw=0, zorder=0)
            ax.add_patch(rect)
        ax.set_frame_on(False)
        ax.tick_params(
            axis='both',
            which='both',
            left=False,
            right=False,
            bottom=False,
            top=False,
            labelbottom=False,
            labelleft=False)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot(self, fig=None):
        """Draw all parts of the plot onto fig or a new figure

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Defaults to a new figure.

        Returns
        -------
        subplots : dict of matplotlib.axes.Axes
            Keys are 'matrix', 'intersections', 'totals', 'shading'
        """
        if fig is None:
            fig = plt.figure(figsize=self._default_figsize)
        specs = self.make_grid(fig)
        shading_ax = fig.add_subplot(specs['shading'])
        self.plot_shading(shading_ax)
        matrix_ax = self._reorient(fig.add_subplot)(specs['matrix'],
                                                    sharey=shading_ax)
        self.plot_matrix(matrix_ax)
        totals_ax = self._reorient(fig.add_subplot)(specs['totals'],
                                                    sharey=matrix_ax)
        self.plot_totals(totals_ax)
        out = {'matrix': matrix_ax,
               'shading': shading_ax,
               'totals': totals_ax}

        for plot in self._subset_plots:
            ax = self._reorient(fig.add_subplot)(specs[plot['id']],
                                                 sharex=matrix_ax)
            if plot['type'] == 'default':
                self.plot_intersections(ax)
            elif plot['type'] in self.PLOT_TYPES:
                kw = plot.copy()
                del kw['type']
                del kw['elements']
                del kw['id']
                self.PLOT_TYPES[plot['type']](self, ax, **kw)
            else:
                raise ValueError('Unknown subset plot type: %r' % plot['type'])
            out[plot['id']] = ax
        return out

    PLOT_TYPES = {
        'catplot': _plot_catplot,
        'stacked_bars': _plot_stacked_bars,
    }

    def _repr_html_(self):
        fig = plt.figure(figsize=self._default_figsize)
        self.plot(fig=fig)
        return fig._repr_html_()


def plot(data, fig=None, **kwargs):
    """Make an UpSet plot of data on fig

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Values for each set to plot.
        Should have multi-index where each level is binary,
        corresponding to set membership.
        If a DataFrame, `sum_over` must be a string or False.
    fig : matplotlib.figure.Figure, optional
        Defaults to a new figure.
    kwargs
        Other arguments for :class:`UpSet`

    Returns
    -------
    subplots : dict of matplotlib.axes.Axes
        Keys are 'matrix', 'intersections', 'totals', 'shading'
    """
    return UpSet(data, **kwargs).plot(fig)

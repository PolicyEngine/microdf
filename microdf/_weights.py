"""Pandas subclass hooks for copying weights and retaining row positions."""

import numpy as np
import pandas as pd


def weight_series(values, index):
    """Own an independent float array aligned positionally to an index."""
    return pd.Series(np.array(values, dtype=float, copy=True), index=index)


def aligned_weights(source, index):
    """Align only when row identity can be established without guessing."""
    if source.index.equals(index):
        return weight_series(source.weights, index)
    if source.index.is_unique:
        positions = source.index.get_indexer(index)
        if (positions >= 0).all():
            return weight_series(source.weights.iloc[positions], index)
    raise ValueError(
        "Cannot propagate weights unambiguously to these rows. "
        "Use positional selection, or explicitly supply weights for the result."
    )


def finalize_weights(result, source, method, previous_weights):
    """Propagate row weights after pandas has finalized its own metadata."""
    if method == "transpose" and result.ndim == 2:
        raise ValueError(
            "Cannot transpose row weights onto columns. "
            "Convert to a pandas DataFrame first and supply explicit result weights."
        )
    if method == "concat":
        objects = source.objs
        if not all(isinstance(obj, WeightPropagationMixin) for obj in objects):
            raise ValueError(
                "Cannot concatenate weighted and unweighted objects: "
                "provide explicit weights for every input."
            )
        weights = concat_weights(result, objects, getattr(source, "axis", None))
        result.weights = weights
        if result.ndim == 2:
            names = [obj.__dict__.get("weights_col") for obj in objects]
            result.weights_col = (
                names[0] if all(name == names[0] for name in names) else None
            )
    elif isinstance(source, WeightPropagationMixin):
        if method == "reset_index" and source.ndim == result.ndim == 1:
            # Series.reset_index(drop=True) changes labels, never row order.
            # pandas 2 constructs a result; pandas 3 relabels a shallow copy.
            result.weights = weight_series(source.weights, result.index)
        elif method == "rename" and previous_weights is not None:
            result.weights = weight_series(previous_weights, result.index)
        else:
            result.weights = aligned_weights(source, result.index)
    return result


def concat_weights(result, objects, axis):
    """Recover concat row identity without relying on pandas 2-only state."""
    if objects[0].ndim == 1 and result.ndim == 2:
        axis = 1
    row_weights = None
    if axis != 1 and len(result) == sum(len(obj) for obj in objects):
        combined_index = objects[0].index.append([obj.index for obj in objects[1:]])
        same_rows = result.index.equals(combined_index)
        reset_rows = result.index.equals(pd.RangeIndex(len(result)))
        keyed_rows = False
        if (
            isinstance(result.index, pd.MultiIndex)
            and result.index.nlevels > combined_index.nlevels
        ):
            trailing = result.index.droplevel(
                list(range(result.index.nlevels - combined_index.nlevels))
            )
            keyed_rows = trailing.equals(combined_index)
        if same_rows or reset_rows or keyed_rows:
            row_weights = np.concatenate([np.asarray(obj.weights) for obj in objects])
    column_weights = None
    if (
        axis != 0
        and result.ndim == 2
        and len(result.columns)
        == sum(obj.shape[1] if obj.ndim == 2 else 1 for obj in objects)
    ):
        values = np.empty(len(result))
        known = np.zeros(len(result), dtype=bool)
        conflict = False
        for obj in objects:
            if obj.index.equals(result.index):
                positions = np.arange(len(result))
            elif obj.index.is_unique:
                positions = obj.index.get_indexer(result.index)
            else:
                conflict = True
                break
            present = positions >= 0
            incoming = np.asarray(obj.weights)[positions[present]]
            overlap = known[present]
            if not np.array_equal(
                values[present][overlap], incoming[overlap], equal_nan=True
            ):
                conflict = True
                break
            values[present] = incoming
            known[present] = True
        if not conflict and known.all():
            column_weights = values
    if row_weights is not None:
        if column_weights is not None and not np.array_equal(
            row_weights, column_weights, equal_nan=True
        ):
            raise ValueError(
                "Ambiguous weights: pandas did not expose the concat axis."
            )
        return weight_series(row_weights, result.index)
    if column_weights is not None:
        return weight_series(column_weights, result.index)
    raise ValueError(
        "Cannot propagate concat weights: conflicting or ambiguous row weights."
    )


class WeightPropagationMixin:
    """Use positional provenance for pandas operations that select rows."""

    def _plain(self):
        return (
            pd.Series(self, copy=False)
            if self.ndim == 1
            else pd.DataFrame(self, copy=False)
        )

    def _weighted_result(self, plain, weights):
        result = type(self)(plain, weights=weight_series(weights, plain.index))
        if self.ndim == 2:
            result.weights_col = self.__dict__.get("weights_col")
        return result

    def _finish_row_operation(self, plain, positions, inplace=False):
        result = self._weighted_result(plain, self.weights.iloc[positions])
        if inplace:
            self._update_inplace(result)
            self.weights = result.weights
            return None
        return result

    def take(self, indices, axis=0, **kwargs):
        """Take values and weights using the same positional indexer."""
        axis = self._get_axis_number(axis)
        plain = self._plain().take(indices, axis=axis, **kwargs)
        weights = self.weights.iloc[indices] if axis == 0 else self.weights
        return self._weighted_result(plain, weights)

    def _slice(self, slobj, axis=0):
        plain = self._plain()._slice(slobj, axis=axis)
        weights = self.weights.iloc[slobj] if axis == 0 else self.weights
        return self._weighted_result(plain, weights)

    def _reindex_with_indexers(
        self, reindexers, fill_value=None, copy=False, allow_dups=False
    ):
        plain = self._plain()._reindex_with_indexers(
            reindexers, fill_value=fill_value, allow_dups=allow_dups
        )
        if copy:
            plain = plain.copy()
        indexer = reindexers.get(0, (None, None))[1]
        if indexer is not None:
            if (np.asarray(indexer) < 0).any():
                raise ValueError(
                    "Cannot invent weights for new rows introduced by reindex."
                )
            weights = self.weights.iloc[indexer]
        else:
            weights = self.weights
        return self._weighted_result(plain, weights)

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        plain = self._plain().drop(
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            errors=errors,
        )
        positions = pd.Series(np.arange(len(self)), index=self.index)
        row_labels = (
            index
            if index is not None
            else labels
            if self._get_axis_number(axis) == 0
            else None
        )
        if row_labels is not None:
            positions = positions.drop(row_labels, level=level, errors=errors)
        return self._finish_row_operation(plain, np.asarray(positions), inplace)

    def sample(self, *args, **kwargs):
        """Sample weights with the selected rows, including replacements."""
        result = super().sample(*args, **kwargs)
        result.weights = weight_series(result.weights, result.index)
        return result

    def sort_values(self, *args, **kwargs):
        """Sort rows and weights together even when labels are duplicated."""
        inplace = kwargs.pop("inplace", False)
        axis = self._get_axis_number(kwargs.get("axis", 0))
        if self.ndim == 2:
            plain = self._plain()
            if axis == 1:
                plain = plain.sort_values(*args, **kwargs)
                return self._finish_row_operation(plain, np.arange(len(self)), inplace)
            marker = object()
            plain = plain.copy(deep=False)
            plain[marker] = np.arange(len(self))
            plain = plain.sort_values(*args, **kwargs)
            positions = np.asarray(plain.pop(marker), dtype=int)
        else:
            values = self._plain()
            positions = pd.Series(
                np.arange(len(self)), index=self.index, name=self.name
            )
            key = kwargs.pop("key", None)
            positions = positions.sort_values(
                *args,
                key=lambda unused: values if key is None else key(values),
                **kwargs,
            )
            plain = values.iloc[np.asarray(positions)]
            plain.index = positions.index
            positions = np.asarray(positions, dtype=int)
        return self._finish_row_operation(plain, positions, inplace)

    def sort_index(self, *args, **kwargs):
        """Sort the index while preserving positional weight provenance."""
        inplace = kwargs.pop("inplace", False)
        axis = self._get_axis_number(kwargs.get("axis", 0))
        if axis == 1:
            plain = self._plain().sort_index(*args, **kwargs)
            return self._finish_row_operation(plain, np.arange(len(self)), inplace)
        positions = pd.Series(np.arange(len(self)), index=self.index).sort_index(
            *args, **kwargs
        )
        plain = self._plain().iloc[np.asarray(positions)]
        plain.index = positions.index
        return self._finish_row_operation(plain, np.asarray(positions), inplace)

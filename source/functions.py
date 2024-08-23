# (C) Copyright 2023 Marco Bauer
# 
# This file is part of polaritonic_adcc.
# 
# polaritonic_adcc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# polaritonic_adcc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with polaritonic_adcc. If not, see <http://www.gnu.org/licenses/>.
#
import libadcc

from qed_amplitude_vec import qed_amplitude_vec
import numpy as np

"""
This function is mostly copied from adcc, because the evaluate and licomb
functions needed to be adapted. This could also be circumvented, if one could
create a libtensor array of specified shape from the python side.
"""

def lincomb(coefficients, tensors, evaluate=False):
    """
    Form a linear combination from a list of tensors.

    If coefficients is a 1D array, just form a single
    linear combination, else return a list of vectors
    representing the linear combination by reading
    the coefficients row-by-row.

    Parameters
    ----------
    coefficients : list
        Coefficients for the linear combination
    tensors : list
        Tensors for the linear combination
    evaluate : bool
        Should the linear combination be evaluated (True) or should just
        a lazy expression be formed (False). Notice that
        `lincomb(..., evaluate=True)`
        is identical to `lincomb(..., evaluate=False).evaluate()`,
        but the former is generally faster.
    """
    if len(tensors) == 0:
        raise ValueError("List of tensors cannot be empty")
    if len(tensors) != len(coefficients):
        raise ValueError("Number of coefficient values does not match "
                         "number of tensors.")
    if isinstance(tensors[0], qed_amplitude_vec):
        return qed_amplitude_vec(**{
            block: lincomb(coefficients, [ten[block] for ten in tensors],
                           evaluate=evaluate)
            for block in tensors[0].keys()
        })
    elif isinstance(tensors[0], np.ndarray):
        return sum(coefficients[i] * gs for i, gs in enumerate(tensors))
    elif not isinstance(tensors[0], libadcc.Tensor):
        raise TypeError("Tensor type not supported")

    if evaluate:
        # Perform strict evaluation on this linear combination
        return libadcc.linear_combination_strict(coefficients, tensors)
    else:
        # Perform lazy evaluation on this linear combination
        start = float(coefficients[0]) * tensors[0]
        return sum((float(c) * t
                    for (c, t) in zip(coefficients[1:], tensors[1:])), start)


def evaluate(a):
    """Force full evaluation of a tensor expression"""
    if isinstance(a, list):
        return [evaluate(elem) for elem in a]
    elif hasattr(a, "evaluate"):
        return a.evaluate()
    else:
        return libadcc.evaluate(a)

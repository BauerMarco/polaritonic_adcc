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
from adcc.AmplitudeVector import AmplitudeVector
import numpy as np

"""
This function is mostly copied from adcc, because the evaluate and licomb
functions needed to be adapted. This could also be circumvented, if one could
create a libtensor array of specified shape from the python side.
"""

class qed_amplitude_vec(AmplitudeVector):
    def __init__(self, **kwargs):
        """
        Construct a polaritonic amplitude vector. Typical use cases are
        ``AmplitudeVector(ph=single_ex_phot_vac, gs1=gs_phot_1, ph1=single_ex_phot_1)``.
        The provided tensor objects should be libtensors for ph, pphh, etc.
        and numpy.ndarrays for gs1, gs2, etc.
        """
        super().__init__(**kwargs)

    def copy(self):
        """Return a copy of the AmplitudeVector"""
        return qed_amplitude_vec(**{k: t.copy() for k, t in self.items()})

    def evaluate(self):
        for k, t in self.items():
            if type(t) == np.ndarray:
                continue
            if type(t) == float:
                self[k] = np.array(t)
                continue
            t.evaluate()
        return self

    @property
    def needs_evaluation(self):
        return any(t.needs_evaluation for k, t in self.items() if type(t) != np.ndarray)

    def dot(self, other):
        """Return the dot product with another AmplitudeVector
        or the dot products with a list of AmplitudeVectors.
        In the latter case a np.ndarray is returned.
        """
        if isinstance(other, list):
            # Make a list where the first index is all singles parts,
            # the second is all doubles parts and so on
            return np.array([float(sum(self[b].dot(av[b]) for b in self.keys())) for av in other])
        else:
            return sum(self[b].dot(other[b]) for b in self.keys())

    def __matmul__(self, other):
        if isinstance(other, qed_amplitude_vec):
            return self.dot(other)
        if isinstance(other, list):
            if all(isinstance(elem, qed_amplitude_vec) for elem in other):
                return self.dot(other)
        return NotImplemented

    def __forward_to_blocks(self, fname, other):
        if isinstance(other, qed_amplitude_vec):
            if sorted(other.keys()) != sorted(self.keys()):
                raise ValueError("Blocks of both qed_amplitude_vec objects "
                                 f"need to agree to perform {fname}")
            ret = {k: getattr(tensor, fname)(other[k])
                   for k, tensor in self.items()}
        else:
            ret = {k: getattr(tensor, fname)(other) for k, tensor in self.items()}
        if any(r == NotImplemented for r in ret.values()):
            return NotImplemented
        else:
            return qed_amplitude_vec(**ret)


    def __mul__(self, other):
        return self.__forward_to_blocks("__mul__", other)

    def __rmul__(self, other):
        return self.__forward_to_blocks("__rmul__", other)

    def __sub__(self, other):
        return self.__forward_to_blocks("__sub__", other)

    def __rsub__(self, other):
        return self.__forward_to_blocks("__rsub__", other)

    def __truediv__(self, other):
        return self.__forward_to_blocks("__truediv__", other)

    def __imul__(self, other):
        return self.__forward_to_blocks("__imul__", other)

    def __iadd__(self, other):
        return self.__forward_to_blocks("__iadd__", other)

    def __isub__(self, other):
        return self.__forward_to_blocks("__isub__", other)

    def __itruediv__(self, other):
        return self.__forward_to_blocks("__itruediv__", other)

    # __add__ is special because we want to be able to add AmplitudeVectors
    # with missing blocks
    def __add__(self, other):
        if isinstance(other, qed_amplitude_vec):
            allblocks = sorted(set(self.keys()).union(other.keys()))
            ret = {k: self.get(k, 0) + other.get(k, 0) for k in allblocks}
            ret = {k: v for k, v in ret.items() if v != 0}
        else:
            ret = {k: tensor + other for k, tensor in self.items()}
        return qed_amplitude_vec(**ret)

    def __radd__(self, other):
        if isinstance(other, qed_amplitude_vec):
            return other.__add__(self)
        else:
            ret = {k: other + tensor for k, tensor in self.items()}
            return qed_amplitude_vec(**ret)

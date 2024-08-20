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
import numpy as np

class diis:
    """Diis helper for qed_ucc2"""
    def __init__(self, max_vec=10):
        self.residuals = []
        self.solutions = []
        self.max_vec = max_vec

    def pop(self):
        if len(self.solutions) > self.max_vec:
            self.solutions.pop(0)
            self.residuals.pop(0)

    def add_vectors(self, solution, residual):
        self.solutions.append(solution)
        self.residuals.append(residual)
        self.pop()

    def get_optimal_linear_combination(self):
        diis_size = len(self.solutions) + 1
        diis_mat = np.zeros((diis_size, diis_size))
        diis_mat[:, 0] = -1.0
        diis_mat[0, :] = -1.0
        for k, r1 in enumerate(self.residuals, 1):
            for ll, r2 in enumerate(self.residuals, 1):
                diis_mat[k, ll] = r1.dot(r2)
                diis_mat[ll, k] = diis_mat[k, ll]
        diis_rhs = np.zeros(diis_size)
        diis_rhs[0] = -1.0
        weights = np.linalg.solve(diis_mat, diis_rhs)[1:]
        solution = 0
        for ii, s in enumerate(self.solutions):
            solution += s * weights[ii]
        return solution.evaluate()

    def do_iteration(self, old, new):
        res = new - old
        rnorm = np.sqrt(res.dot(res))
        self.add_vectors(new, res)
        #t2 = t2new
        if len(self.solutions) > 2 and rnorm <= 1.0:
            new = self.get_optimal_linear_combination()
            diff = new - self.solutions[-1]
            diff.evaluate()
            rnorm = np.sqrt(diff.dot(diff))
        return new, rnorm

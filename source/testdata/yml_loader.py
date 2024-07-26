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
import os
import yaml

def read_yaml_data(fname):
    thisdir = os.path.dirname(__file__)
    yaml_file = os.path.join(thisdir, fname)
    with open(yaml_file, "r") as f:
        data_raw = yaml.safe_load(f)
    data = data_raw.copy()
    for key in data:
        data[key] = np.array(data[key])
    return data


qed_energies_psi4 = read_yaml_data("testdata_psi4.yml")


"""
Copyright 2020, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import subprocess


def get_taskset():
    # example output: pid 29502's current affinity list: 0,4
    try:
        cpus = subprocess.check_output(['taskset', '-cp', str(os.getpid())]).decode()
        return cpus.split(':')[-1].strip()
    except FileNotFoundError:  # taskset not available on system
        return None

def get_taskset_num_CPUs():
    """ Get total number of CPUs. Note that overlap is not handled. """
    taskset = get_taskset()
    total = 0
    for entry in taskset.split(','):
        if '-' in entry:
            lo, hi = map(int, entry.split('-'))
            total += hi - lo + 1
        else:
            total += 1
    return total


def test_get_taskset_num_CPUs():
    assert get_taskset_num_CPUs('0,4') == 2
    assert get_taskset_num_CPUs('0,4,7-8') == 4
    assert get_taskset_num_CPUs('0,4-16,7-8') == 16


def get_num_GPUs():
    return int(subprocess.check_output('nvidia-smi -L | wc -l', shell=True).decode())


def get_num_CPUs():
    return int(subprocess.check_output('cat /proc/cpuinfo | grep processor | wc -l', shell=True).decode())


def main():
    print(get_taskset())
    print(get_num_GPUs())
    print(get_num_CPUs())


if __name__ == '__main__':
    main()

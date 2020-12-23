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
# Task array helper for the Sun Grid Engine.
# Needs adaptation for other clusters.

import argparse
import pickle
import subprocess
import time

import fasteners
import os


def _env_get_int(key, default):
    try:
        return int(os.environ.get(key, default))
    except ValueError:
        return default


JOB_ID =  _env_get_int('JOB_ID', 0)
FIRST =   _env_get_int('SGE_TASK_FIRST', 0)  # >= 1 if set
LAST =    _env_get_int('SGE_TASK_LAST', 0)
STEP =    _env_get_int('SGE_TASK_STEP', 1)
TASK_ID = _env_get_int('SGE_TASK_ID', 1) - 1  # in [0 ... LAST-1]

assert STEP == 1, STEP  # currently not supported

NUM_TASKS = LAST - FIRST + 1


# TODO: have some join function, wait for all NUM_TASKS to join, then return result to each?


def get_running_tasks(job_id):
    otp = subprocess.check_output(['qstat']).decode()
    return sum(1 for l in otp.splitlines() if l.startswith(job_id))


def post(k, v):
    if JOB_ID == 0:
        return
    with fasteners.InterProcessLock('.lock.{}'.format(JOB_ID)):
        o, p = _read(JOB_ID)
        o[k] = v
        with open(p, 'wb') as f:
            pickle.dump(o, f)


def post_dict_update(k, d):
    if JOB_ID == 0:
        return
    with fasteners.InterProcessLock('.lock.{}'.format(JOB_ID)):
        o, p = _read(JOB_ID)
        if k not in o: # set
            o[k] = d
        else:  # update
            o[k].update(d)
        with open(p, 'wb') as f:
            pickle.dump(o, f)


def _read(job_id):
    p = '.pickle.{}'.format(job_id)
    if not os.path.isfile(p):
        return {}, p
    else:
        with open(p, 'rb') as f:
            return pickle.load(f), p


def join(job_id, verbose=False):
    while True:
        num_running = get_running_tasks(job_id)
        if verbose:
            print('\r{: 3d} running'.format(num_running), end='')
        if num_running == 0:
            if verbose:
                print()
            break
        time.sleep(1)
    o, p = _read(job_id)
    return o, p


def job_enumerate(it):
    for i, el in enumerate(it):
        if i % NUM_TASKS == TASK_ID:
            yield i, el



def main():
    p = argparse.ArgumentParser()
    p.add_argument('-j')
    flags = p.parse_args()
    if flags.j:
        print(Accumulator().join(flags.j))
        return

    import time
    out_dir = 'testing_task_array'
    os.makedirs(out_dir, exist_ok=True)
    for i, el in job_enumerate(range(100)):
        with open(os.path.join(out_dir, str(i)), 'w') as f:
            f.write(str(TASK_ID) + '\n')
        Accumulator().post(str(i), el)


if __name__ == '__main__':
    main()

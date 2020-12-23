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

--------------------------------------------------------------------------------

Some notes
- only git fetches if within .git
"""

import os
import shutil
import subprocess
from socket import gethostname

from fjcommon.assertions import assert_exc
from fjcommon.qsuba_git_helper import unique_checkout


from helpers.config_checker import DEFAULT_CONFIG_DIR


# TODO: check



class ConfigsRepo(object):
    def __init__(self,
                 config_dir=DEFAULT_CONFIG_DIR,  # just set to 'configs' usually
                 git_url='git@gitlab.com:fab-jul/jointcomp_configs.git',
                 git_checkout='origin/master',
                 defer_setup=False):
        self.config_dir = config_dir
        self.git_url = git_url
        self.git_checkout = git_checkout
        self._did_setup = False
        if not defer_setup:
            self.setup()

    def fetch_and_check(self, *config_ps):
        unique_checkout(self.config_dir, self.git_url, self.git_checkout)
        self._did_setup = True
        print(self._did_setup)
        self.check_configs_available(*config_ps, _retry=False)

    def setup(self):
        """
        The configs repository is not part of the main repo (jointcomp).
        If we are running from a git checkout, the current file will be under a git repo at LSTM_COMP_ROOT_GIT. In this case,
        LSTM_COMP_ROOT_GIT is assumed to be unique per experiment (via qsuba_git_helper.py, which uses $CUDA_VISIBLE_DEVICES).
        If we are in LSTM_COMP_ROOT_GIT, checkout configs under LSTM_COMP_ROOT_GIT/src/configs, which is $CWD/configs
        Otherwise, assume configs repo exists in $CWD/configs
        """
        git_root = _get_git_root()  # where .git is located
        if git_root is not None:  # we are in a git pulled environment, update configs repo
            if 'narigpu' in gethostname():
                print('In narigpu, not pulling configs repo...')
            else:
                print('Updating configs repo in {}...'.format(self.config_dir))
                unique_checkout(self.config_dir, self.git_url, self.git_checkout)
        _assert_dir_exists_in_cwd(self.config_dir)
        self._did_setup = True
        return self

    def check_configs_available(self, *config_ps, _retry=True):
        """
        checks if all config_ps are available. Only pull if a config is not found.
        config_ps are interpreted as relative to CWD, just like self.config_dir!
        """
        assert self._did_setup
        for p in config_ps:
            # check if self.configs_dir is contained in path
            assert p.startswith(self.config_dir), f'Expected {p} to start with {self.config_dir}!'
            if not os.path.isfile(p):
                # unexpected, as we just did a fetch. Most likely, file was not committed, but we do one retry
                # (somewhat unmotivated though)
                print(f'*** File not found: {p}. In {os.getcwd()}, status:')
                print(_get_current_git_status(self.config_dir))
                if _is_git():
                    print(_get_current_git_status(self.config_dir))
                    if _retry:
                        print('Trying to fetch again...')
                        self.setup()
                        self.check_configs_available(*config_ps, _retry=False)
                raise ValueError(f'Config not found: {p}')

    def force_pull(self):
        _git_pull(self.config_dir)


def _assert_in_jointcomp_root(check_for='run_test.py'):
    assert check_for in os.listdir(os.getcwd()), 'Assuming we run from jointcomp/src, but no {} found in {}: {}'.format(
            check_for, os.getcwd(), os.listdir(os.getcwd()))


def _assert_dir_exists_in_cwd(p):
    assert_exc(os.path.isdir(p), '{} not in {}'.format(p, os.getcwd()))


def _git_pull(configs_repo_p, show_err=False):
    print('git pull {}'.format(configs_repo_p))
    ret = subprocess.call(['git', 'pull'], cwd=configs_repo_p, stderr=subprocess.DEVNULL if not show_err else None)
    return ret == 0


def _is_git():
    """ Check if cwd is part of a git repo """
    git_commit = subprocess.call(['git', 'rev-parse', 'HEAD'],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
    return int(git_commit) == 0  # 0: success, 128: error


def _get_current_git_status(git_repo_p):
    return subprocess.check_output(['git', 'show'], cwd=git_repo_p).decode()


def _get_git_root():
    """ Return path to folder in which .git is located """
    if not _is_git():
        return None
    cwd = os.getcwd()  # is absolute
    assert cwd[0] == os.path.sep
    for i in range(10):
        if cwd == os.path.sep:
            break
        if '.git' in os.listdir(cwd):
            return cwd
        cwd = os.path.dirname(cwd)
    raise ValueError('No .git found!')


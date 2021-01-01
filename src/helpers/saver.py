"""
Copyright 2020, ETH Zurich

This file is part of RC-PyTorch.

RC-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

RC-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RC-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
import os
import time
import re
import shutil

import pytorch_ext as pe
from os.path import basename
import torch
from torch.optim import optimizer
from fjcommon.no_op import NoOp
from fjcommon import timer
from fjcommon.assertions import assert_exc

class CkeckpointLoadingException(Exception):
    pass




class _CheckpointTracker(object):
    """ out_dir is usally set via set_out_dir """
    def __init__(self, out_dir=None, ckpt_name_fmt='ckpt_{:010d}.pt', tmp_postfix='.tmp'):
        assert len(tmp_postfix)
        assert '.' in tmp_postfix
        m = re.search(r'{:0(\d+?)d}', ckpt_name_fmt)
        assert m, 'Expected ckpt_name_fmt to have an int specifier such as or {:09d} or {:010d}.'
        max_itr = 10 ** int(m.group(1)) - 1
        if max_itr < 10000000:  # ten million, should be enough
            print(f'Maximum iteration supported: {max_itr}')
        assert os.sep not in ckpt_name_fmt
        self.ckpt_name_fmt = ckpt_name_fmt
        self.ckpt_prefix = ckpt_name_fmt.split('{')[0]
        assert len(self.ckpt_prefix), 'Expected ckpt_name_fmt to start with a prefix before the format part!'
        self.tmp_postfix = tmp_postfix

        self._out_dir = None
        if out_dir is not None:
            self.set_out_dir(out_dir)

    def set_out_dir(self, out_dir):
        assert self._out_dir is None
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir

    def get_all_ckpts(self):
        """
        :return: All checkpoints in `self._out_dir`, sorted ascendingly by global_step.
        """
        return [os.path.join(self._out_dir, f)
                for f in sorted(os.listdir(self._out_dir))
                if f.startswith(self.ckpt_prefix)]

    def itr_ckpt(self):
        for ckpt_p in self.get_all_ckpts():
            yield self.get_itr_from_ckpt_p(ckpt_p), ckpt_p

    def get_ckpt_for_itr(self, itr, before_time=None):
        """
        Gets ckpt_itrc where itrc <= itr, i.e., the latest ckpt before `itr`.
        If `before_time` is given and itr == -1, the latest ckpt before `before_time` is used
        Special values: itr == -1 -> newest ckpt
        """
        # sorted list of (itr, ckpt_p)
        ckpts = list(self.itr_ckpt())
        if before_time is not None:
            ckpts_before_time = [(i, p) for i, p in ckpts if os.path.getmtime(p) <= before_time]
            print(f'*** Ignoring {len(ckpts) - len(ckpts_before_time)} ckpts after {before_time}')
            ckpts = ckpts_before_time
        assert_exc(len(ckpts) > 0, 'No ckpts found in {}'.format(self._out_dir), CkeckpointLoadingException)
        if itr == -1:
            return ckpts[-1]
        first_itrc, _ = ckpts[0]
        assert_exc(first_itrc <= itr, 'Earliest ckpt {} is after {}'.format(first_itrc, itr), CkeckpointLoadingException)
        for itrc, ckpt_p in reversed(ckpts):
            if itrc <= itr:
                return itrc, ckpt_p
        raise ValueError('Unexpected, {}, {}'.format(itr, ckpts))

    def get_latest_ckpt(self):
        """
        :return: Most recent checkpoint. May be a temporary checkpoint.
        """
        try:
            return self.get_all_ckpts()[-1]
        except IndexError:
            raise ValueError('No checkpoints found in', self._out_dir)

    def get_lastest_persistent_ckpt(self):
        """
        :return: Most recent persistent checkpoint. May be a temporary checkpoint.
        """
        candidates = [p for p in self.get_all_ckpts() if not p.endswith(self.tmp_postfix)]
        if len(candidates) == 0:
            raise ValueError('No persistent checkpoints')
        return candidates[-1]

    def _get_out_p(self, global_step, is_tmp):
        postfix = self.tmp_postfix if is_tmp else ''
        # put it as prefix as the base class does prefix matching -> we don't want this temporary ckpt to match!
        name = self.ckpt_name_fmt.format(global_step) + postfix
        return (os.path.join(self._out_dir, n) for n in (name, 'saving_' + name))

    def get_itr_from_ckpt_p(self, ckpt_p):
        file_name = os.path.splitext(os.path.basename(ckpt_p))[0]
        assert self.ckpt_prefix in file_name
        itr_part = file_name.replace(self.ckpt_prefix, '')
        itr_part_digits_only = int(''.join(c for c in itr_part if c.isdigit()))
        return itr_part_digits_only



class Saver(_CheckpointTracker):
    """
    Saves ckpts:
    - ckpt_XXXXXXXX.pt.tmp
    If keep_tmp_last=None:
        Every `keep_every`-th ckpt is renamed to
        - ckpt_XXXXXXXX.pt
        and kept, the intermediate ones are removed. We call this a persistent checkpoint.
    else:
        Let C be the most recent persistent checkpoint.
        In addition to C being kept, the last `keep_tmp_last` temporary checkpoints before C are also kept.
        This means that always `keep_tmp_last` more checkpoints are kept than if keep_tmp_last=None
    """
    def __init__(self,
                 keep_tmp_itr: int, keep_every=10, keep_tmp_last=None,
                 out_dir=None, ckpt_name_fmt='ckpt_{:010d}.pt', tmp_postfix='.tmp',
                 verbose=False):
        """
        :param keep_every: keep every `keep_every`-th checkpoint, making it a persistent checkpoint.
        :param keep_tmp_itr: keep checkpoint every `keep_tmp_itr` iterations.
        :param keep_tmp_last: Also keep the last `keep_tmp_last` temporary checkpoints before a persistent checkpoint.
        :param ckpt_name_fmt: filename, must include a format spec and some prefix before the format
        :param tmp_postfix: non-empty string to append to temporary checkpoints
        :param verbose: if True, print rename and remove info.
        """
        self.keep_every = keep_every
        self.keep_tmp_last = keep_tmp_last
        self.keep_tmp_itr = keep_tmp_itr
        self.ckpts_since_last_permanent = 0
        self.print = print if verbose else NoOp
        self.save_time_acc = timer.TimeAccumulator()
        super(Saver, self).__init__(out_dir, ckpt_name_fmt, tmp_postfix)

    def save(self, modules, global_step, force=False, make_permanent=False):
        """
        Save iff (force given or global_step % keep_tmp_itr == 0)
        :param modules: dictionary name -> nn.Module
        :param global_step: current step
        :return: bool, Whether previous checkpoints were removed
        """
        if not (force or (global_step % self.keep_tmp_itr == 0)):
            return False
        assert self._out_dir is not None
        current_ckpt_p = self._save(modules, global_step)
        self.ckpts_since_last_permanent += 1
        if make_permanent or (self.ckpts_since_last_permanent == self.keep_every):
            self._remove_previous_and_make_permanent(current_ckpt_p)
            self.ckpts_since_last_permanent = 0
            return True
        return False

    def _save(self, modules, global_step):
        out_p, out_p_wip = self._get_out_p(global_step, is_tmp=True)
        with self.save_time_acc.execute():
            torch.save({key: m.state_dict() for key, m in modules.items()}, out_p_wip)
            os.rename(out_p_wip, out_p)
        return out_p

    def _remove_previous_and_make_permanent(self, current_ckpt_p):
        assert self.tmp_postfix in current_ckpt_p
        current_ckpt_p_non_tmp = current_ckpt_p.replace(self.tmp_postfix, '')
        self.print('{} -> {}'.format(basename(current_ckpt_p), basename(current_ckpt_p_non_tmp)))
        os.rename(current_ckpt_p, current_ckpt_p_non_tmp)
        keep_tmp_last = self.get_all_ckpts()[-(self.keep_tmp_last+1):] if self.keep_tmp_last else []
        for p in self.get_all_ckpts():
            if self.tmp_postfix in p and p not in keep_tmp_last:
                self.print('Removing {}...'.format(basename(p)))
                os.remove(p)
        self.print('Average save time: {:.3f}s'.format(self.save_time_acc.mean_time_spent()))


class Restorer(_CheckpointTracker):
    def restore_latest_persistent(self, net):
        return self.restore(net, self.get_lastest_persistent_ckpt())

    def restore(self, modules, ckpt_p, strict=True, restore_restart=False):
        print('Restoring {}... (strict={})'.format(ckpt_p, strict))
        map_location = None if pe.CUDA_AVAILABLE else 'cpu'
        state_dicts = torch.load(ckpt_p, map_location=map_location)
        # ---
        # For quick and dirty ckpt re-use, where strict=False is not sufficient
        # (e.g. same variable name, different shape)
        ignore_keys = os.environ.get('IGNORE_KEYS', None)
        if ignore_keys:
            ignore_keys = ignore_keys.split(':')
        for key, m in modules.items():
            # optim implements its own load_state_dict which does not have the `strict` keyword...
            if isinstance(m, optimizer.Optimizer):
                if restore_restart:
                    print('Not restoring optimizer, --restore_restart given...')
                elif ignore_keys and 'optim' in ignore_keys:
                    print('*** Ignoring optim')
                else:
                    try:
                        m.load_state_dict(state_dicts[key])
                    except ValueError as e:
                        print('Error while restoring Optimizer:', str(e))
            else:
                try:
                    d = state_dicts[key]

                    if ignore_keys:
                        print(ignore_keys)
                        keys_to_remove = set()
                        for k in ignore_keys:
                            for k_d in d:
                                if k in k_d:
                                    keys_to_remove.add(k_d)
                        print('Ignoring', keys_to_remove)
                        for k_d in keys_to_remove:
                            d.pop(k_d)
                        
                    m.load_state_dict(d, strict=strict)
                except RuntimeError as e:  # loading error
                    raise e
        return self.get_itr_from_ckpt_p(ckpt_p)


def _convert_fuckup(state_dicts):
    """
    odict_keys(['head.0.weight', 'head.0.bias', 'head.3.weight', 'head.3.bias', 'model.0.body.0.weight',
    'model.0.body.0.bias', 'model.0.body.3.weight', 'model.0.body.3.bias', 'model.1.body.0.weight',
     'model.1.body.0.bias', 'model.1.body.3.weight', 'model.1.body.3.bias', 'model.2.body.0.weight',
     'model.2.body.0.bias', 'model.2.body.3.weight', 'model.2.body.3.bias', 'model.3.body.0.weight',
     'model.3.body.0.bias', 'model.3.body.3.weight', 'model.3.body.3.bias', 'model.4.body.0.weight',
     'model.4.body.0.bias', 'model.4.body.3.weight', 'model.4.body.3.bias', 'model.5.body.0.weight',
     'model.5.body.0.bias', 'model.5.body.3.weight', 'model.5.body.3.bias', 'model.6.body.0.weight',
     'model.6.body.0.bias', 'model.6.body.3.weight', 'model.6.body.3.bias', 'model.7.body.0.weight',
     'model.7.body.0.bias', 'model.7.body.3.weight', 'model.7.body.3.bias', 'model.8.weight',
     'model.8.bias', 'model.9.body.0.weight', 'model.9.body.0.bias', 'model.9.body.3.weight',
      'model.9.body.3.bias', 'model.10.body.0.weight', 'model.10.body.0.bias', 'model.10.body.3.weight',
       'model.10.body.3.bias', 'model.11.body.0.weight', 'model.11.body.0.bias', 'model.11.body.3.weight',
        'model.11.body.3.bias', 'model.12.body.0.weight', 'model.12.body.0.bias', 'model.12.body.3.weight',
         'model.12.body.3.bias', 'model.13.body.0.weight', 'model.13.body.0.bias', 'model.13.body.3.weight',
          'model.13.body.3.bias', 'model.14.body.0.weight', 'model.14.body.0.bias', 'model.14.body.3.weight',
          'model.14.body.3.bias', 'model.15.body.0.weight', 'model.15.body.0.bias', 'model.15.body.3.weight',
           'model.15.body.3.bias', 'model.16.body.0.weight', 'model.16.body.0.bias', 'model.16.body.3.weight',
            'model.16.body.3.bias', 'tail.0.weight', 'tail.0.bias'])
    :param state_dicts:
    :return:
    """
    import re
    if 'net' not in state_dicts or 'model.16.body.3.bias' not in state_dicts['net']:
        return state_dicts
    net = state_dicts['net']
    net_out = {}
    for k in net.keys():
        if k.startswith('model.'):
            m = re.search('model.(\d+)', k)
            assert m, k
            idx = int(m.group(1))
            if idx < 4:
                net_out[k] = net[k]
            elif 4 <= idx < 8:
                print('Skipping', idx)
            elif 8 <= idx < 12:
                k_out = k.replace(str(idx), str(idx - 4))
                print(k, '0>', k_out)
                net_out[k_out] = net[k]
            else:
                print('Skipping', idx)
        else:
            net_out[k] = net[k]
    print(net_out.keys())
    state_dicts['net'] = net_out
    exit(1)
    return state_dicts






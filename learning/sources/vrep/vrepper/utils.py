import functools
import warnings
import subprocess as sp
import os

import psutil

from vrepper.lib.vrepConst import simx_opmode_oneshot, simx_opmode_blocking, simx_return_ok

list_of_instances = []
import atexit


def cleanup():  # kill all spawned subprocesses on exit
    for i in list_of_instances:
        i.end()


atexit.register(cleanup)

blocking = simx_opmode_blocking
oneshot = simx_opmode_oneshot


# the class holding a subprocess instance.
class instance():
    def __init__(self, args, suppress_output=True):
        self.args = args
        self.suppress_output = suppress_output
        list_of_instances.append(self)

    def start(self):
        print('(instance) starting...')
        try:
            if self.suppress_output:
                stdout = open(os.devnull, 'w')
            else:
                stdout = sp.STDOUT
            self.inst = sp.Popen(self.args, stdout=stdout, stderr=sp.STDOUT)
        except EnvironmentError:
            print('(instance) Error: cannot find executable at', self.args[0])
            raise

        return self

    def isAlive(self):
        return True if self.inst.poll() is None else False

    def end(self):
        print('(instance) terminating...')
        if self.isAlive():
            pid = self.inst.pid
            parent = psutil.Process(pid)
            for _ in parent.children(recursive=True):
                _.kill()
            retcode = parent.kill()
        else:
            retcode = self.inst.returncode
        print('(instance) retcode:', retcode)
        return self


# check return tuple, raise error if retcode is not OK,
# return remaining data otherwise
def check_ret(ret_tuple, ignore_one=False):
    istuple = isinstance(ret_tuple, tuple)
    if not istuple:
        ret = ret_tuple
    else:
        ret = ret_tuple[0]

    if (not ignore_one and ret != simx_return_ok) or (ignore_one and ret > 1):
        raise RuntimeError('retcode(' + str(ret) + ') not OK, API call failed. Check the paramters!')

    return ret_tuple[1:] if istuple else None


def deprecated(msg=''):
    def dep(func):
        '''This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.'''

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.warn_explicit(
                "Call to deprecated function {}. {}".format(func.__name__, msg),
                category=DeprecationWarning,
                filename=func.func_code.co_filename,
                lineno=func.func_code.co_firstlineno + 1
            )
            return func(*args, **kwargs)

        return new_func

    return deprecated

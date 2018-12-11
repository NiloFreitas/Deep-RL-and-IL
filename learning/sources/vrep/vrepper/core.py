# V-REP as tethered robotics simulation environment
# Python Wrapper
# Qin Yongliang 20170410

# import the vrep library
from vrepper.lib import vrep
from vrepper.lib.vrepConst import sim_handle_all, simx_headeroffset_server_state, \
    sim_scripttype_childscript

from inspect import getargspec
import types
import numpy as np
import socket
from contextlib import closing

from vrepper.utils import check_ret, blocking, oneshot, instance, deprecated
from vrepper.vrep_object import vrepobject


class vrepper(object):
    """ class holding a v-rep simulation environment and
    allowing to call all the V-Rep remote functions ("simx..")
    """

    def __init__(self, port_num=None, dir_vrep='', headless=False, suppress_output=True):
        if port_num is None:
            port_num = self.find_free_port_to_use()

        self.port_num = port_num

        if dir_vrep == '':
            print('(vrepper) trying to find V-REP executable in your PATH')
            import distutils.spawn as dsp
            path_vrep = dsp.find_executable('vrep.sh')  # fix for linux
            if path_vrep == None:
                path_vrep = dsp.find_executable('vrep')
        else:
            path_vrep = dir_vrep + 'vrep'
        print('(vrepper) path to your V-REP executable is:', path_vrep)
        if path_vrep is None:
            raise Exception("Sorry I couldn't find V-Rep binary. "
                            "Please make sure it's in the PATH environmental variable")

        # start V-REP in a sub process
        # vrep.exe -gREMOTEAPISERVERSERVICE_PORT_DEBUG_PREENABLESYNC
        # where PORT -> 19997, DEBUG -> FALSE, PREENABLESYNC -> TRUE
        # by default the server will start at 19997,
        # use the -g argument if you want to start the server on a different port.
        args = [path_vrep, '-gREMOTEAPISERVERSERVICE_' + str(self.port_num) + '_FALSE_TRUE']

        if headless:
            args.append('-h')

        # instance created but not started.
        self.instance = instance(args, suppress_output)

        self.cid = -1
        # clientID of the instance when connected to server,
        # to differentiate between instances in the driver

        self.started = False

        # is the simulation currently running (as far as we know)
        self.sim_running = False

        # assign every API function call from vrep to self
        vrep_methods = [a for a in dir(vrep) if
                        not a.startswith('__') and isinstance(getattr(vrep, a), types.FunctionType)]

        def assign_from_vrep_to_self(name):
            wrapee = getattr(vrep, name)
            arg0 = getargspec(wrapee)[0][0]
            if arg0 == 'clientID':
                def func(*args, **kwargs):
                    return wrapee(self.cid, *args, **kwargs)
            else:
                def func(*args, **kwargs):
                    return wrapee(*args, **kwargs)
            setattr(self, name, func)

        for name in vrep_methods:
            assign_from_vrep_to_self(name)

    def find_free_port_to_use(
            self):  # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    # start everything
    def start(self):
        if self.started == True:
            raise RuntimeError('you should not call start() more than once')

        print('(vrepper) starting an instance of V-REP...')
        self.instance.start()

        # try to connect to V-REP instance via socket
        retries = 0
        while True:
            print('(vrepper) trying to connect to server on port', self.port_num, 'retry:', retries)
            # vrep.simxFinish(-1) # just in case, close all opened connections
            self.cid = self.simxStart(
                '127.0.0.1', self.port_num,
                waitUntilConnected=True,
                doNotReconnectOnceDisconnected=True,
                timeOutInMs=1000,
                commThreadCycleInMs=0)  # Connect to V-REP

            if self.cid != -1:
                print('(vrepper) Connected to remote API server!')
                break
            else:
                retries += 1
                if retries > 15:
                    self.end()
                    raise RuntimeError('(vrepper) Unable to connect to V-REP after 15 retries.')

        # Now try to retrieve data in a blocking fashion (i.e. a service call):
        objs, = check_ret(self.simxGetObjects(
            sim_handle_all,
            blocking))

        print('(vrepper) Number of objects in the scene: ', len(objs))

        # Now send some data to V-REP in a non-blocking fashion:
        self.simxAddStatusbarMessage(
            '(vrepper)Hello V-REP!',
            oneshot)

        # setup a useless signal
        self.simxSetIntegerSignal('asdf', 1, blocking)

        print('(vrepper) V-REP instance started, remote API connection created. Everything seems to be ready.')

        self.started = True
        return self

    # kill everything, clean up
    def end(self):
        print('(vrepper) shutting things down...')
        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        # vrep.simxGetPingTime(clientID)

        # Now close the connection to V-REP:
        if self.sim_running:
            self.stop_simulation()
        self.simxFinish()
        self.instance.end()
        print('(vrepper) everything shut down.')
        return self

    def load_scene(self, fullpathname):
        print('(vrepper) loading scene from', fullpathname)
        try:
            check_ret(self.simxLoadScene(fullpathname,
                                         0,  # assume file is at server side
                                         blocking))
        except:
            print('(vrepper) scene loading failure')
            raise
        print('(vrepper) scene successfully loaded')

    def start_blocking_simulation(self):
        self.start_simulation(True)

    def start_nonblocking_simulation(self):
        self.start_simulation(False)

    def start_simulation(self, is_sync):
        # IMPORTANT
        # you should poll the server state to make sure
        # the simulation completely stops before starting a new one
        while True:
            # poll the useless signal (to receive a message from server)
            check_ret(self.simxGetIntegerSignal(
                'asdf', blocking))

            # check server state (within the received message)
            e = self.simxGetInMessageInfo(
                simx_headeroffset_server_state)

            # check bit0
            not_stopped = e[1] & 1

            if not not_stopped:
                break

        # enter sync mode
        check_ret(self.simxSynchronous(is_sync))
        check_ret(self.simxStartSimulation(blocking))
        self.sim_running = True

    def make_simulation_synchronous(self, sync):
        if not self.sim_running:
            print('(vrepper) simulation doesn\'t seem to be running. starting up')
            self.start_simulation(sync)
        else:
            check_ret(self.simxSynchronous(sync))

    def stop_simulation(self):
        check_ret(self.simxStopSimulation(oneshot), ignore_one=True)
        self.sim_running = False

    @deprecated('Please use method "stop_simulation" instead.')
    def stop_blocking_simulation(self):
        self.stop_simulation()

    def step_blocking_simulation(self):
        check_ret(self.simxSynchronousTrigger())

    def get_object_handle(self, name):
        handle, = check_ret(self.simxGetObjectHandle(name, blocking))
        return handle

    def get_object_by_handle(self, handle, is_joint=True):
        """
        Get the vrep object for a given handle

        :param int handle: handle code
        :param bool is_joint: True if the object is a joint that can be moved
        :returns: vrepobject
        """
        return vrepobject(self, handle, is_joint)

    def get_object_by_name(self, name, is_joint=True):
        """
        Get the vrep object for a given name

        :param str name: name of the object
        :param bool is_joint: True if the object is a joint that can be moved
        :returns: vrepobject
        """
        return self.get_object_by_handle(self.get_object_handle(name), is_joint)

    @staticmethod
    def create_params(ints=[], floats=[], strings=[], bytes=''):
        if bytes == '':
            bytes_in = bytearray()
        else:
            bytes_in = bytes
        return (ints, floats, strings, bytes_in)

    def call_script_function(self, function_name, params, script_name="remoteApiCommandServer"):
        """
        Calls a function in a script that is mounted as child in the scene

        :param str script_name: the name of the script that contains the function
        :param str function_name: the name of the function to call
        :param tuple params: the parameters to call the function with (must be 4 parameters: list of integers, list of floats, list of string, and bytearray

        :returns: tuple (res_ints, res_floats, res_strs, res_bytes)
            WHERE
            list res_ints is a list of integer results
            list res_floats is a list of floating point results
            list res_strs is a list of string results
            bytearray res_bytes is a bytearray containing the resulting bytes
        """
        assert type(params) is tuple
        assert len(params) == 4

        return check_ret(self.simxCallScriptFunction(
            script_name,
            sim_scripttype_childscript,
            function_name,
            params[0],  # integers
            params[1],  # floats
            params[2],  # strings
            params[3],  # bytes
            blocking
        ))

    def get_global_variable(self, name, is_first_time):
        if is_first_time:
            return vrep.simxGetFloatSignal(self.cid, name, vrep.simx_opmode_streaming)
        else:
            return vrep.simxGetFloatSignal(self.cid, name, vrep.simx_opmode_buffer)

    def _convert_byte_image_to_color(self, res, img):
        reds = np.zeros(res[0] * res[1], dtype=np.uint8)
        greens = np.zeros(res[0] * res[1], dtype=np.uint8)
        blues = np.zeros(res[0] * res[1], dtype=np.uint8)
        for i in range(0, len(img), 3):
            reds[int(i / 3)] = img[i] & 255
            greens[int(i / 3)] = img[i + 1] & 255
            blues[int(i / 3)] = img[i + 2] & 255

        img_out = np.zeros((res[0], res[1], 3), dtype=np.uint8)
        img_out[:, :, 0] = np.array(reds).reshape(res)
        img_out[:, :, 1] = np.array(greens).reshape(res)
        img_out[:, :, 2] = np.array(blues).reshape(res)

        return img_out

    def get_image(self, object_id):
        res, img = check_ret(self.simxGetVisionSensorImage(object_id, 0, blocking))
        return self._convert_byte_image_to_color(res, img)

    @staticmethod
    def flip180(image):
        return np.rot90(image, 2, (0, 1))

    def _convert_depth_to_image(self, res, depth):
        reshaped_scaled = 255 - np.array(depth).reshape(res) * 255  # because is in range [0,1] and inverted
        rounded = np.around(reshaped_scaled, 0).astype(np.uint8)
        return rounded

    def _convert_depth_to_rgb(self, res, depth):
        rounded = self._convert_depth_to_image(res, depth)
        img = np.zeros((res[0], res[1], 3), dtype=np.uint8)
        img[:, :, 0] = rounded
        img[:, :, 1] = rounded
        img[:, :, 2] = rounded
        return img

    def get_depth_image(self, object_id):
        res, depth = check_ret(self.simxGetVisionSensorDepthBuffer(object_id, blocking))
        return self._convert_depth_to_image(res, depth)

    def get_depth_image_as_rgb(self, object_id):
        res, depth = check_ret(self.simxGetVisionSensorDepthBuffer(object_id, blocking))
        return self._convert_depth_to_rgb(res, depth)

    def get_image_and_depth(self, object_id):
        img = self.get_image(object_id)
        depth = self.get_depth_image(object_id)

        out = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        out[:, :, :3] = img
        out[:, :, 3] = depth

        return out

    def get_collision_handle(self, name_of_collision_obj):
        """ In order to use this you first have to open the scene in V-Rep, then
            click on "calculation module properties" on the left side (the button
            that looks like "f(x)"), then click "add new collision object", chose the
            two things between which you want to check for collision (one of them can be a collection
            which you can create in yet another window), and finally double click on the new
            collision object in order to rename it to something more catchy than "Collision".
            You can find more information here:
            http://www.coppeliarobotics.com/helpFiles/en/collisionDetection.htm
            Also don't forget to save the scene after adding the collision object.

        :param name_of_collision_obj: the "#" is added automatically at the end
        :return: collision_handle (this is an integer that you need for check_collision)
        """
        return check_ret(self.simxGetCollisionHandle(name_of_collision_obj + "#", blocking))[0]

    def check_collision(self, collision_handle):
        """ At any point in time call this function to get a boolean value if the
            collision object is currently detecting a collision. True for collision.

        :param collision_handle: integer, the handle that you obtaind from
                                    "get_collision_handle(name_of_collision_obj)"
        :return: boolean
        """
        return check_ret(self.simxReadCollision(collision_handle, blocking))[0]

    def get_collision_object(self, name_of_collision_obj):
        """ this is effectively the same as "get_collision_handle" but instead of an
            integer (the handle) it instead returns an object that has a ".is_colliding()"
            function, which is super marginally more convenient.

        :param name_of_collision_obj: string, name of the collision object in V-Rep
        :return: Collision object that you can check with ".is_colliding()->bool"
        """

        handle = check_ret(self.simxGetCollisionHandle(name_of_collision_obj + "#", blocking))[0]
        col = Collision(env=self, handle=handle)
        return col


class Collision(object):
    def __init__(self, env, handle):
        self.handle = handle
        self.env = env

    def is_colliding(self):
        return check_ret(self.env.simxReadCollision(self.handle, blocking))[0]

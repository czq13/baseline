import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import os
#os.environ['GAZEBO_PLUGIN_PATH'] = '/home/czq/chWorkspace/tailsitter_env/fw/gazebo_model'
os.environ['GAZEBO_PLUGIN_PATH'] = '/home/czq/firmware/build/posix_sitl_default/build_gazebo'
os.environ['GAZEBO_MODEL_PATH'] = '/home/czq/eclipse-workspace/tailsitter_env/fw/gazebo_model'
#os.environ['LD_LIBRARY_PATH'] = '/home/czq/chWorkspace/tailsitter_env/fw/gazebo_model'
os.environ['LD_LIBRARY_PATH'] = '/home/czq/firmware/build/posix_sitl_default/build_gazebo'
#import tailsitter_env as te
import baselines.TailsitterEnv.tailsitter_env as te
import random
class TailsitterEnv(gym.Env):
    count = 0
    def __init__(self):
        self.min_action = -1.0#np.array([-1.0, -1.0])
        self.max_action = 1.0#np.array([1.0, 1.0])
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32)
        self.low_state = -1.0#np.array([-1.0, -1.0, -1.0])
        self.high_state = 1.0#np.array([1.0, 1.0, 1.0])
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=(3,) ,dtype=np.float32)
        self.num = random.randint(10, 100)
        os.environ['GAZEBO_MASTER_URI'] = 'http://localhost:115' + str(self.num)
        print(os.environ['GAZEBO_MASTER_URI'])
        te.init()
        TailsitterEnv.count += 1
        self.seed()

    def seed(self,seed=None):
        self.np_random,seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        taction = [0] * 2
        taction[0], taction[1] = (action[0]+1.0) / 2.0, -1.0 * (action[1]+1.0) / 2.0 * 1.57
        te.SetCtrl(taction[0], taction[1])
        obsv, reward, done = self.get_obsv()
        return obsv, reward, done, {}


    def reset(self):
        s = [0.0] * 6
        noise = np.array([self.np_random.uniform(low=-1, high=1), 0])
        s[0] *= 0.5
        s[1] *= 0.5
        s[2] *= 0.5
        s[3] *= 5.0 / 57.3
        s[4] *= 5.0 / 57.3
        s[5] *= 5.0 / 57.3
        te.ResetEnv(self.num, s[0],s[1],s[2],s[3],s[4],s[5])
        obsv, reward, done = self.get_obsv()
        return obsv

    def get_obsv(self):
        tobsv = te.GetObsv()
        vx, vz, pitch, roll, pitch_speed, z = tobsv[0], tobsv[1], tobsv[2], tobsv[3], tobsv[4], tobsv[5]
        obsv = np.array([vx / 30.0, vz / 10.0, pitch / (1.57)])
        #reward = -pitch / 10.0 + (vx - 15.0) / 100.0
        reward = 0.0
        #if pitch > -1.57 + 10.0/57.3:
        reward += (min(-pitch,1.57-10.0/57.3)) / 100.0
        if math.fabs(vx) < 15.0:
            reward += (math.fabs(vx) - 15.0) / 1000.0
        done = False

        if vz < -5:
            reward, done = -1.0, True
            return obsv, reward, done
        #elif vz > 2.0:
        #    reward -= (vz - 2.0) / 100.0
        if vz < 0:
            reward -= (min(math.fabs(vz), 1.0))/1000
        if vz < -1:
            reward -= (min(math.fabs(vz+1), 1.0))/200
        if vz < -2:
            reward -= (min(math.fabs(vz+2), 1.0))/100
        if vz < -3:
            reward -= (math.fabs(vz+3)) / 50
        if vz > 5:
            reward -= (math.fabs(vz-5)) / 10000
        if math.fabs(vx) > 18.0 and pitch < -1.57 + 5.0/57.3:
            reward, done = reward + 1.0, True
            return obsv, reward, done

        if math.fabs(pitch_speed) > 1.0:
            reward -= min((math.fabs(pitch_speed) - 1.0), 1.0) / 400.0
        if math.fabs(pitch_speed) > 2.0:
            reward -= min((math.fabs(pitch_speed) - 2.0), 1.0) / 200.0
        if math.fabs(pitch_speed) > 3.0:
            reward -= min((math.fabs(pitch_speed) - 3.0), 1.0) / 100.0

        return obsv, reward, done

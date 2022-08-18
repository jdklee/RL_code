import numpy as np
from utils.test_env import EnvTest
import random
import gym


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon        = eps_begin
        self.eps_begin      = eps_begin
        self.eps_end        = eps_end
        self.nsteps         = nsteps


    def update(self, t):
        """
        Updates epsilon

        Args:
            t: int
                frame number

          modifies self.epsilon such that
          it is a linear interpolation from self.eps_begin to
          self.eps_end as t goes from 0 to self.nsteps
          For t > self.nsteps self.epsilon remains constant
        """
        value = np.linspace(self.eps_end, self.eps_begin, int(self.nsteps)+1)

        self.epsilon = value[t] if t <= self.nsteps else self.eps_end




class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)


    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int
                best action according some policy
        Returns:
            an action
        """
        """
        Functionality:
                with probability self.epsilon, return a random action
                else, return best_action. Access env via self.env
                Generate a random action using self.env.action_space.sample()
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return best_action


def test1():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)

    found_diff = False
    for i in range(10):
        rnd_act = exp_strat.get_action(0)
        if rnd_act != 0 and rnd_act is not None:
            found_diff = True

    assert found_diff, "Test 1 failed."
    print("Test1: ok")


def test2():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0, 10)
    exp_strat.update(5)
    assert exp_strat.epsilon == 0.5, "Test 2 failed"
    print("Test2: ok")


def test3():
    env = EnvTest((5, 5, 1))
    exp_strat = LinearExploration(env, 1, 0.5, 10)
    exp_strat.update(20)
    assert exp_strat.epsilon == 0.5, "Test 3 failed"
    print("Test3: ok")



if __name__ == "__main__":
    test1()
    test2()
    test3()

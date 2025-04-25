import copy
import gym
import pickle

def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False

def add_clone_methods_to_env(env):
    def clone_state():
        safe_dict = {}
        for k, v in env.__dict__.items():
            if is_picklable(v):
                safe_dict[k] = v
        return copy.deepcopy(safe_dict)

    def restore_state(state):
        for k, v in state.items():
            env.__dict__[k] = copy.deepcopy(v)

    env.clone_state = clone_state
    env.restore_state = restore_state
    return env

def patch_env():
    original_make = gym.make

    def new_make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        return add_clone_methods_to_env(env)

    gym.make = new_make
    gym.Env.clone_state = add_clone_methods_to_env(gym.Env).clone_state
# encoding: utf-8
'''
Q_learing sample: Multi agent learning.
'''
import numpy as np
import random

step = 100 # 100 is enough number for optimization, but 10 is not enough.
lr = 0.1
gan = 0.9

'''
actions: dictionary of action index and moving of action.
'''
actions = {
    0: -1, #west
    1: 1 #east
}

'''
status: dictionary of status name and that's index.
'''
status = {
    "bandit":0,
    "lake":1,
    "entrance":2,
    "forest":3,
    "treasure":4,
}

'''
rewards: dictionary of reward each status.
'''
rewards = {
    0: -1,
    1: 0,
    2: 0,
    3: 0,
    4: 1
}

'''
q_table: matrix of (status) x (actions).
'''
q_table = np.zeros((len(status),len(actions)), dtype=np.float32)

def select_action(s):
    action = np.argmax(q_table[s])

    # monte carlo
    if random.random() < 0.9:
        action = random.randint(0, 1)
    return action


def update_q(s, a, s_n, r):
    q_table[s, a] = (1 - lr) * q_table[s, a] + lr * (r + gan * (np.max(q_table[s_n])))


def train(verbose=False):
    print("actions: %s" % (actions,))
    print("status: %s" % (status,))
    print("q_table: %s" % q_table)

    # simulation loop
    for i in xrange(step):
        # decide initial status
        s = status["entrance"]
        if verbose:
            print("initial s: %s" % (s))

        history = []
        end = False
        while(end is not True):
            # select action
            a = select_action(s)
            if verbose:
                print("select action: %s" % (a))
            history.append((s, a))

            # check status of end(if we are on end status, we don't need to move next status.)
            if verbose:
                print("status s: %s" % s)
            if status["bandit"] == s or status["treasure"] == s:
                if verbose:
                    print("end!!")
                reward = rewards[s]
                if verbose:
                    print("reward: %s" % (reward))
                    print("history: %s" % (history))
                current_index = len(history) - 1
                previous_index = len(history) - 2
                update_q(history[previous_index][0], history[previous_index][1], history[current_index][0], reward)
                if verbose:
                    print("update q_table: %s" % q_table)
                break

            # update q
            s_n = s + actions[a] # next_status
            reward = rewards[s_n] # status -> next_status's reward
            update_q(s, a, s_n, reward) # update using next reward
            s = s_n
    print("="*100)
    print("final q_table: %s" % (q_table))


if __name__ == '__main__':
    train(verbose=False)

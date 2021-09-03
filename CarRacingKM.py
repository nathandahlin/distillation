import numpy as np

import itertools as it
from skimage import color, transform
from sklearn.tree import DecisionTreeClassifier

class CarRacingKM(DecisionTreeClassifier):
    """
    CarRacing specific part of the SoftBinaryDecisionTree

    Some minor env-specifig tweaks but overall
    assumes very little knowledge from the environment
    """

    def __init__(self, clf, scaler, max_negative_rewards=100, **kwargs):
        all_actions = np.array(
            [k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])]
        )
        # car racing env gives wrong pictures without render
        #kwargs["render"] = True
        #super().__init__(
            #action_map=all_actions,
            #pic_size=(96, 96),
        #    **kwargs
        #)
        self.action_map = all_actions
        self.clf = clf
        self.scaler = scaler

        self.gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in all_actions])
        self.break_actions = np.array([a[2] == 1 for a in all_actions])
        self.n_gas_actions = self.gas_actions.sum()
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_negative_rewards

    @staticmethod
    def process_image(obs):
        return 2 * color.rgb2gray(obs) - 1.0

    #def get_random_action(self):
    #    """
    #    Here random actions prefer gas to break
    #    otherwise the car can never go anywhere.
    #    """
    #    action_weights = 14.0 * self.gas_actions + 1.0
    #    action_weights /= np.sum(action_weights)

    #    return np.random.choice(self.dim_actions, p=action_weights)

    def check_early_stop(self, reward, totalreward):
        if reward < 0:
            self.neg_reward_counter += 1
            #print('neg rewards: '+str(self.neg_reward_counter))
            done = (self.neg_reward_counter > self.max_neg_rewards)

            if done and totalreward <= 500:
                punishment = -20.0
            else:
                punishment = 0.0
            if done:
                self.neg_reward_counter = 0

            return done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0
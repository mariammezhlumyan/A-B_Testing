import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger

Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """
    Visualization class to analyze the learning behavior of 
    Epsilon-Greedy and Thompson Sampling algorithms.

    Attributes:
        df (pd.DataFrame): Combined DataFrame of rewards and actions.
    """

    def __init__(self, df):
        self.df = df

    def plot1(self):
        """
        Plot cumulative reward over trials for each algorithm.

        This plot helps visualize how each algorithm learns and accumulates reward over time.
        """
        df_copy = self.df.copy()
        df_copy["Trial"] = df_copy.groupby("Algorithm").cumcount()
        df_grouped = df_copy.groupby(["Algorithm", "Trial"])["Reward"].mean().unstack(level=0)
        df_grouped.cumsum().plot(title="Cumulative Reward by Algorithm")

        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.show()

    def plot2(self):
        """
        Plot cumulative rewards and cumulative regrets for each algorithm.

        This plot compares both algorithms in terms of total reward and regret over trials.
        """

        df = self.df.copy()
        df["Best_Reward"] = max(Bandit_Reward)
        df["Cumulative_Reward"] = df.groupby("Algorithm")["Reward"].cumsum()
        df["Cumulative_Regret"] = df.groupby("Algorithm").cumcount() * max(Bandit_Reward) - df["Cumulative_Reward"]

        # Plot for Cumulative Rewards
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        for algo in df["Algorithm"].unique():
            subset = df[df["Algorithm"] == algo]
            plt.plot(subset["Cumulative_Reward"].values, label=algo)
        plt.title("Cumulative Rewards")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)

        # Plot for Cumulative Regrets
        plt.subplot(1, 2, 2)
        for algo in df["Algorithm"].unique():
            subset = df[df["Algorithm"] == algo]
            plt.plot(subset["Cumulative_Regret"].values, label=algo)
        plt.title("Cumulative Regret")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm implementation.

    Attributes:
        p (list): True reward values for each bandit.
        n_arms (int): Number of bandit arms.
        counts (list): Number of times each arm was selected.
        values (list): Estimated mean reward of each arm.
        total_reward (float): Total accumulated reward.
        epsilon (float): Exploration parameter.
        trials (int): Number of trials.
        rewards (list): List of rewards received.
        actions (list): List of selected arms.
    """

    def __init__(self, p, epsilon=1.0):
        self.p = p
        self.n_arms = len(p)
        self.counts = [0] * self.n_arms
        self.values = [0.0] * self.n_arms
        self.total_reward = 0
        self.epsilon = epsilon
        self.trials = NumberOfTrials
        self.rewards = []
        self.actions = []

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon})"

    def pull(self, t):
        """
        Select an arm based on the epsilon-greedy strategy.

        Args:
            t (int): Current trial number.

        Returns:
            int: Index of selected arm.
        """
        epsilon = self.epsilon / (t + 1)
        if random.random() < epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """
        Update estimates after receiving a reward.

        Args:
            chosen_arm (int): Index of selected arm.
            reward (float): Received reward.
        """

        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] += (reward - value) / n
        self.total_reward += reward
        self.rewards.append(reward)
        self.actions.append(chosen_arm)

    def experiment(self):
        """
        Run the Epsilon-Greedy experiment over all trials.
        """
        for t in range(self.trials):
            arm = self.pull(t)
            reward = np.random.normal(self.p[arm], 1.0)
            self.update(arm, reward)

    def report(self):
        """
        Generate report for Epsilon-Greedy.

        Saves results to CSV and logs average reward and cumulative regret.

        Returns:
            pd.DataFrame: DataFrame containing actions and rewards.
        """
        df = pd.DataFrame({
            "Bandit": self.actions,
            "Reward": self.rewards,
            "Algorithm": "EpsilonGreedy"
        })
        df.to_csv("epsilon_greedy_results.csv", index=False)
        avg_reward = np.mean(self.rewards)
        regret = sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"E-Greedy Avg Reward: {avg_reward:.4f}")
        logger.info(f"E-Greedy Cumulative Regret: {regret:.2f}")
        return df

#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling algorithm implementation.

    Attributes:
        p (list): True reward values for each bandit.
        n_arms (int): Number of bandit arms.
        alpha (list): Alpha parameters of Beta distributions.
        beta (list): Beta parameters of Beta distributions.
        trials (int): Number of trials.
        rewards (list): List of rewards received.
        actions (list): List of selected arms.
        total_reward (float): Total accumulated reward.
    """
    def __init__(self, p):
        self.p = p
        self.n_arms = len(p)
        self.alpha = [1] * self.n_arms
        self.beta = [1] * self.n_arms
        self.trials = NumberOfTrials
        self.rewards = []
        self.actions = []
        self.total_reward = 0

    def __repr__(self):
        return "ThompsonSampling()"

    def pull(self, _):
        """
        Select an arm based on Thompson Sampling strategy.

        Returns:
            int: Index of selected arm.
        """

        sampled = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return np.argmax(sampled)

    def update(self, chosen_arm, reward):
        """
        Update Beta distribution parameters after receiving a reward.

        Args:
            chosen_arm (int): Index of selected arm.
            reward (float): Received reward.
        """
        if reward > 0:
            self.alpha[chosen_arm] += 1
        else:
            self.beta[chosen_arm] += 1
        self.rewards.append(reward)
        self.actions.append(chosen_arm)
        self.total_reward += reward

    def experiment(self):
        """
        Run the Thompson Sampling experiment over all trials.
        """
        for t in range(self.trials):
            arm = self.pull(t)
            reward = np.random.normal(self.p[arm], 1.0)
            self.update(arm, reward)

    def report(self):
        """
        Generate report for Thompson Sampling.

        Saves results to CSV and logs average reward and cumulative regret.

        Returns:
            pd.DataFrame: DataFrame containing actions and rewards.
        """
        df = pd.DataFrame({
            "Bandit": self.actions,
            "Reward": self.rewards,
            "Algorithm": "ThompsonSampling"
        })
        df.to_csv("thompson_sampling_results.csv", index=False)
        avg_reward = np.mean(self.rewards)
        regret = sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"Thompson Sampling Avg Reward: {avg_reward:.4f}")
        logger.info(f"Thompson Sampling Cumulative Regret: {regret:.2f}")
        return df


def comparison():
    """
    Run and compare Epsilon-Greedy and Thompson Sampling experiments.

    Executes experiments, generates reports, saves combined results, 
    and visualizes cumulative rewards and regrets.
    """
    eg = EpsilonGreedy(Bandit_Reward)
    eg.experiment()
    df_eg = eg.report()

    ts = ThompsonSampling(Bandit_Reward)
    ts.experiment()
    df_ts = ts.report()

    df_combined = pd.concat([df_eg, df_ts], ignore_index=True)
    df_combined.to_csv("combined_results.csv", index=False)

    viz = Visualization(df_combined)
    viz.plot1()
    viz.plot2()

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

print(comparison())

"""Imitation learning helpers — collect demonstrations and behavior-clone the policy.

Used to bootstrap MAPPO from a non-random initial policy, breaking the
entropy-lock that occurs when reward signal is too sparse for random
actions to ever achieve.
"""

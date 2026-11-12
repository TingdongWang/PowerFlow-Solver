"""Sufficient statistics for the full-data squared-error training objective.
For input X and target Y, minimize ||X A - Y||_F^2 + lambda ||A||_F^2.
This module is used only by the offline training optimizer.
"""

def regression_statistics(x,y):
    return x.T@x,x.T@y

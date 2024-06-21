# On this cell, I just import all stuff that is needed or probably needed in the future
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import math


# This is the function for scipy odeint that helps it to solve the diff equation
# Differential equation


def generate_plot(gamma, method):
    def helper(t, theta):
        theta, z = theta
        dthetadt = [z, f * math.cos(Omega * t) - omegaSQ * math.sin(theta) - lamda * z]
        return dthetadt

    # Parameters
    Omega = 2 * np.pi
    omega = Omega * 1.5
    lamda = omega / 2
    omegaSQ = omega ** 2
    f = gamma * omegaSQ

    t = np.linspace(0, 50, 5001)
    initial_conditions = [0, 0]
    solutions_list = []

    # Loop over different rtol and atol values
    for i in range(-6, -13, -1):  # goes from 1e-6 to 1e-12
        print("On iteration ", i)
        # Solve the differential equation
        sol = solve_ivp(helper, [0, 50], initial_conditions, t_eval=t, method=method, rtol=10 ** (i),
                        atol=10 ** (i - 2))
        solutions_list.append(sol.y[0])

    # Find the maximum length of the solutions
    max_length = max(len(solution) for solution in solutions_list)

    # Pad shorter solutions with NaN or zeros to make them the same length
    padded_solutions_list = []
    for solution in solutions_list:
        padded_solution = np.pad(solution, (0, max_length - len(solution)), 'constant', constant_values=np.nan)
        padded_solutions_list.append(padded_solution)

    # Convert to a numpy array
    solutions_array = np.array(padded_solutions_list)

    # Calculate the standard deviation of solutions_list
    std_dev = np.nanstd(solutions_array, axis=0)  # Use nanstd to ignore NaNs in the calculation

    # Plotting
    plt.figure(figsize=(12, 10))
    for i, solution in enumerate(solutions_array):
        plt.plot(t[:len(solution)], (solution * 180) / (np.pi), label=f"1e{-6 - i}")

    # Plot the standard deviation
    plt.plot(t[:len(std_dev)], (std_dev * 180) / (np.pi), label='Standard Deviation', color='black', linestyle='--')

    plt.xlabel('t')
    plt.ylabel('phi(t) (rotations)')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"Graphs/{method}_{gamma:.2f}".replace('.', '_') + ".png")



gamma_array = np.arange(0.80, 1.30 + 0.01, 0.01)
model_array = [
    "RK45",
    "BDF",
    "Radau",
    "LSODA",
    "RK23",
    "DOP853"]

for model in model_array:
    for gamma in gamma_array:
        generate_plot(gamma, model)


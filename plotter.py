# import matplotlib.pyplot as plt

# alphas = [1.0, 0.5, 0.1, 0.001]

# results = {
#     "bayesian": [
#         3.775931656027584,
#         3.1140593211790137,
#         2.626372264562719,
#         2.469497810893133,
#     ],
#     "robust_A2C": [
#         2.53095373295087,
#         2.480473340072817,
#         2.4836447656705363,
#         2.4843059493354103,
#     ],
#     "robust_A2C_with_principal": [
#         2.5016456368195055,
#         2.495525758062742,
#         2.514559589975584,
#         2.472150797229394,
#     ],
# }

# # plot the results
# for model, values in results.items():
#     plt.plot(alphas, values, label=model)

# # add title and labels
# plt.title("Model performance with different alpha values")
# plt.xlabel("Alpha")
# plt.ylabel("Performance")

# # add legend
# plt.legend()

# # display the plot
# plt.show()

import matplotlib.pyplot as plt

# Data from the tables
alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

bayesian_agent_values = [2.402, 2.513, 2.830, 3.114, 3.466, 3.732]
oblivious_robust_agent_a2c_values = [2.514, 2.532, 2.513, 2.506, 2.514, 2.527]
oblivious_robust_agent_ppo_values = [2.509, 2.527, 2.534, 2.489, 2.524, 2.508]

mindful_robust_agent_a2c_values = [2.584, 2.601, 2.550, 2.518, 2.525, 2.504]
mindful_robust_agent_ppo_values = [2.559, 2.620, 2.567, 2.535, 2.519, 2.497]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(alpha_values, bayesian_agent_values, label="BayesianAgent", color="black")
plt.plot(
    alpha_values,
    oblivious_robust_agent_a2c_values,
    label="ObliviousRobustAgentA2C",
    color="red",
)
plt.plot(
    alpha_values,
    oblivious_robust_agent_ppo_values,
    label="ObliviousRobustAgentPPO",
    color="green",
)

plt.plot(
    alpha_values,
    mindful_robust_agent_a2c_values,
    label="MindfulRobustAgentA2C",
    color="magenta",
)
plt.plot(
    alpha_values,
    mindful_robust_agent_ppo_values,
    label="MindfulRobustAgentPPO",
    color="blue",
)

# Add some labels and a legend
plt.xlabel("Alpha")
plt.ylabel("Values")
plt.title("Bayesian Agent vs Oblivious Robust Agent vs Mindful Robust Agent")
plt.legend()

# Show the plot
plt.show()

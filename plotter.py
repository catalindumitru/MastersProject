import matplotlib.pyplot as plt

alphas = [1.0, 0.5, 0.1, 0.001]

results = {
    "bayesian": [
        3.775931656027584,
        3.1140593211790137,
        2.626372264562719,
        2.469497810893133,
    ],
    "robust_A2C": [
        2.53095373295087,
        2.480473340072817,
        2.4836447656705363,
        2.4843059493354103,
    ],
    "robust_A2C_with_principal": [
        2.5016456368195055,
        2.495525758062742,
        2.514559589975584,
        2.472150797229394,
    ],
}

# plot the results
for model, values in results.items():
    plt.plot(alphas, values, label=model)

# add title and labels
plt.title("Model performance with different alpha values")
plt.xlabel("Alpha")
plt.ylabel("Performance")

# add legend
plt.legend()

# display the plot
plt.show()

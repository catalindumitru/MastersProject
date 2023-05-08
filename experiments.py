import numpy as np
from game import Game
from pickle import dump

if __name__ == "__main__":
    iterations = 4
    alphas = [0.1, 0.05, 0.01]
    results = {}

    for alpha in alphas:
        print("Alpha:", alpha)
        results[alpha] = {}
        results_alpha = []
        for it in range(iterations):
            print(it)
            while True:
                try:
                    game = Game(alpha)
                    game_results = game.play()
                except:
                    continue
                else:
                    break
            results_alpha.append(game_results)

        for key in Game.keys:
            values = [results_alpha[i][key] for i in range(iterations)]
            results[alpha][key] = np.median(values)

        print(results[alpha])
        print()

    with open("results1.pkl", "wb") as f:
        dump(results, f)

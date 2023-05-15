import numpy as np
from game import Game
from pickle import dump


class Experiments:
    iterations = 10
    alphas = [1, 0.8, 0.6, 0.4, 0.2, 0]

    def run(self):
        results = {}
        for alpha in self.alphas:
            print("Alpha:", alpha)
            results[alpha] = {}
            results_alpha = []
            for it in range(self.iterations):
                print("  Iteration:", it)
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
                values = [results_alpha[i][key] for i in range(self.iterations)]
                results[alpha][key] = np.median(values)

            print(results[alpha])
            print()

        with open("results.pkl", "wb") as f:
            dump(results, f)

# The try-except block is used to avoid potential errors caused by gurobipy.
# Because the space dimensions of the environment are very large, 
# it is possible that the solver cannot find a solution in a reasonable time
# and times out. In this case, the game is restarted with a fresh environment.


if __name__ == "__main__":
    experiments = Experiments()
    experiments.run()

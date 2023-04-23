import subprocess
import sys
METRIC = sys.argv[1]

PREFIX = ['python', 'main.py']
for K in ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '30', '50']:
    for N in ['32', '64', '128', '256']:
        # for METRIC in ['None', 'LFDA', 'MLKR', 'NCA']:
        for METHOD in ['Manhattan', 'Euclidean']:
            subprocess.run(PREFIX + [
                f'--method={METHOD}',
                f'--metric_learning={METRIC}',
                f'--k_neighbors={K}',
                f'--n_components={N}'
            ]
            )

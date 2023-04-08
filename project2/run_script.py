import subprocess

PREFIX = ['python', 'main.py']
for K in ['5', '10', '20', '50']:
    for METRIC in ['None', 'MLKR','LFDA','NCA', ]:
        for METHOD in ['Manhattan', 'Euclidean']:
            subprocess.run(PREFIX + [
                f'--method={METHOD}',
                f'--metric_learning={METRIC}',
                f'--k_neighbors={K}'
            ]
                           )

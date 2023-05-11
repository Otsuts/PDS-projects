import subprocess
import sys


PREFIX = ['python', 'main.py']
for MODEL in ['semantic_relatedness', 'semantic_embedding']:
    for LR in ['5e-4', '1e-4', '5e-5', '1e-5', '5e-6', '1e-6']:
        subprocess.run(PREFIX+[
            f'--model={MODEL}',
            f'--learning_rate={LR}',
            '--use_big'
        ])
        subprocess.run(PREFIX+[
            f'--model={MODEL}',
            f'--learning_rate={LR}',
        ])

for LR in ['5e-4', '1e-4', '5e-5', '1e-5', '5e-6', '1e-6']:
    for GLR in ['1e-6', '1e-5']:
        for DLR in ['1e-5', '1e-4']:
            for NS in ['1', '400', '1000', '2000']:
                subprocess.run(PREFIX + [
                    '--model=synthetic',
                    f'--learning_rate={LR}',
                    f'--g_lr={GLR}',
                    f'--d_lr={DLR}',
                    f'--num_samples{NS}'
                ])

import os
import sys

from lib.config import TrainConfig

if __name__ == '__main__':
    config_path = sys.argv[1]
    fname = sys.argv[2]
    config = TrainConfig.from_json(config_path)
    args = [
        'lib',
        'configs',
        'server.py',
        'reformat_data.py',
        'metadata.json',
        config.implicit.model_file,
        config.catboost.model_file,
        config.products_enriched_file,
        config.implicit.vectors_file,
    ]
    args_str = ' '.join(args)

    cmd = f'zip -r {fname} {args_str}'
    os.system(cmd)

"""结果输出"""

import json
import os

from absl import app


def main(argv):
  del argv
  base_name = 'experiment'
  base_dir = '/tmp/model_dir'
  for name in ['DSA', 'activity']:
    print(name)
    for seed in [1, 2, 3]:
      model_dir = os.path.join(base_dir, name, f'{base_name}_seed_{seed}')
      fit_dir = os.path.join(model_dir, 'fit', 'results.json')
      with open(fit_dir, 'r') as fp:
        results = json.load(fp)
      print(results)

if __name__ == '__main__':
  app.run(main)

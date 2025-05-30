#!/bin/bash
virtualenv sequential_attention
source sequential_attention/bin/activate
pip install --editable .

# get data
#sh Two_Stage/experiments/get_all_data.sh

# run experiments
python -m Two_Stage.experiments.run --num_epochs=200 --data_name=activity

#!/bin/bash





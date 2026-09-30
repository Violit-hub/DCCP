"""Set the Coffee demonstration dataset path in the retained environment config."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', required=True)
args = parser.parse_args()
dataset = Path(args.dataset).expanduser().resolve()
if not dataset.is_file():
    parser.error('dataset must be an existing HDF5 file')
root = Path(__file__).resolve().parents[1]
config = root / 'assets/coffee/data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json'
data = json.loads(config.read_text(encoding='utf-8'))
data['train']['data'] = str(dataset)
with config.open('w', encoding='utf-8', newline='\n') as handle:
    json.dump(data, handle, indent=2)
    handle.write('\n')
print('Updated Coffee dataset config:', config)

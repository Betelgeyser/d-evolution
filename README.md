# D-evolution
D-evolution is a simple neuroevolutional software, once started as a hobby project to learn more about machine learning and neural networks in particular.

## Features and limitations
* All computions are performed on CUDA GPUs to achieve better performance
* Training is implemented using evolutionary algorithms
* Supports multivariate regression
* Currently only feed-forward networks are supported
* Currently sutable only for regression tasks
* Currently there is no simple way to use trained model

## Documentation
Documentation is in the code itself, though sometimes may lack.
To build the documentation you can simply run `dub -b docs` and it will be generated and stored in the `docs` directory.

## Usage example
	./d-evolution --data="path/to/data" -d 0 --min -100 --max 100 -n 10 -l 3 -t 60 --training -r 100 -p 100 -f="MAPE" --output-file="result.json"
* `--data` - path to data directory. It must contain `training` and `validation` directories with training and validation datasets respectively. Each of this directories must contain `inputs.csv` and `outputs.csv` files.
* `-d, --device` - CUDA device to use. Defaults to `0`
* `-t, --time` - time limit in seconds
* `-l, --layers` - number of layers in a network
* `-n, --neurons` - number of neurons in every layer
* `-p, --population` - population size
* `-r, --report` - print training report every X generations
* `--training` - run in training mode
* `-f, --fitness-function` - fitness function to use. Available values: MAE (default), MAPE
* `--min` - minimum connection weight
* `--max` - maximum connection weight
* `-s, --seed` - seed for the PRNG. May be set to a specific unsigned integer number or `random` (default)
* `--output-file` - output file to save trained results

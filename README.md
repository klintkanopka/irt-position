# irt-position

# Overview

Fits an IRT mixture model that models item position effects. Requires `torch`, `numpy`, and `pandas`

# Data

Data files should be organized with one row per response and have four columns:

- **id** - a unique id associatd with the respondent
- **itemkey** - a unique id associated with each item
- **sequence_number** - the position (1-indexed) that the respondent encounters the item in
- **resp** - a dichotomously coded response indicator, coded 0 if incorrect and 1 if correct

Included is a 500 student, 44 item data file: `data/tiny_test.csv`

# Usage

Run `main.py` from the command line. If this full repo is cloned, it will function with no additional arguments. For help with arguments, run `python main.py -h`

Implemented arguments are as follows:

- **--input**: Input file path. Defaults to `data/tiny_test.csv'
- **--irt_model**: Number of parameters for the underlying IRT model. Specifying `1` will implement a mixture of Rasch models and specifying `2` will implement a mixture of 2PLs. Defaults to `2`
- **--epochs**: Maximum number of training epochs. One epoch is a full E step followed by a full M step. Currently there is no early stopping at the epoch-level implemented. Defaults to 100 epochs
- **--max_iterations**: Maximum number of iterations within an individual E or M step. One iteration is a full pass through the data. Early stopping is implemented here. Defaults to 100 iterations
- **--batch_size**: Batch size for the stochastic gradient descent optimizer. Defaults to 128
- **--num_workers**: Number of parallel workers for data loader. Specifying more workers than your machine has cores may be sub-optimal. Defaults to 2
- **--learning_rate_E**: Learning rate for E step. Defaults to 0.1
- **--learning_rate_M**: Learning rate for M step. Defaults to 0.01
- **--epsilon**: Epsilon used for convergence criterion. Note loss is currently implemented as total negative log likelihood, so scaling epsilon with data size is advised. Defaults to 0.01
- **--verbose**: Prints additional fitting information and running loss within E and M steps. Defaults to `False`


# Output

Upon loading data, two additional files are created, with base file names corresponding to the original data file used:

- **filename\_person\_key.csv** relates rows of the person parameter file (called `sid` internally) to the original respondent ids
- **filename\_item\_key.csv** relates rows of the item paramter file (called `ik` internally) to the original item ids

After fitting the model, creates two additional files, with base file names corresponding to the original data file used:


- **filename\_person\_params.csv** contains fitted person parameters. Columns are: `id`, `sid`, `theta`, `k`, `c`
- **filename\_item\_params.csv** contains fitted item parameters. Columns are: `itemkey`, `ik`, `beta_e`, `beta_l`, `alpha_e`, `alpha_l`

Person-side parameters are:

- `id`: Respondent identifier from original dataset
- `sid`: Internal, 0-indexed, respondent key
- `theta`: Estimated ability
- `k`: Mixing location parameter. Specifies the item position where the mixture is half early, half late. Not constrained to be an integer
- `c`: Mixing scale parameter. Controls the slope of the mixing curve. Larger values represent more abrupt transitions from early to late item response functions

Item-side parameters are:

- `itemkey`: Item identifier from original dataset
- `ik`: Internal, 0-indexed, item key
- `beta_e`: Early-test item difficulty. Note that to idenitfy the model, these are forced to sum to zero during fitting. After fitting, check to see that this is approximately true (it should be) - if not, demean this column and let me know
- `beta_l`: Late-test item difficulty
- `alpha_e`: Early-test item discrimination. If `--irt-model 1` is specified, this column will still be returned, but populated with all ones
- `alpha_l`: Late-test item discrimination. If `--irt-model 1` is specified, this column will still be returned, but populated with all ones

# Misc

- Automatically uses GPU for computation if available. If this causes problems, try reducing batch size.

# To do:

- Flags to force CPU/GPU usage
- Clean up early stopping implementation
- Improve command line diagnostic printing (using `tqdm`)
- Add `tensorboard` integration

# tradeoff_between_domain_alignment_and_reconstruction_loss
This code is from paper "Trade-off between reconstruction loss and feature alignment for domain generalization". 

####### A. For CS-CMNIST dataset only!

First, you need to install some package for satisfying all the requirements below:

torch 1.6.0

torchvision 0.7.0

numpy 1.19.1

tqdm 4.41.1

Second, please take a look at the tested algorithms in "algorithm.py". The algorithms are denoted from #1 to # 4 in order of IRM + reconstruction loss, CEM + reconstruction loss, IRM-MMD + reconstruction loss, IB-IRM + reconstruction loss. To keep the same number of hyper-parameters and utilize the existing source code, all 04 algorithms are under the same name of "IBIRM". Please select only one algorithm from these 04 algorithms to run at a time and comment out other ones. In the default code, we are using "IRM + reconstruction loss" algorithm and commented out other ones.

Third, please run "sweep_train.py" to get the result. "sweep_train.py" scans over the algorithms and the hyper-parameters. The setting of hyper-parameters can be found at the end of "sweep_train.py". Please select the algorithm you want to sweep and comment out other ones.

In addition, there are some parameters that can be selected in "sweep_train.py". For example, --test_val to control the tuning procedures, --env_seed to select the oracle model, --holdout_fraction to control the validation set. For now, we use the default values of 5% data for validation and use the train-validation tuning procedure. You do not need to adjust these parameters.

The checkpoints are not stored, and the final results will be printed after the whole run. Our code is based on this repo at https://github.com/ahujak/IB-IRM.


####### B. For CMNIST dataset only!

First, you need to install some package for satisfying the below requirements:

numpy==1.20.3

wilds==1.2.2

imageio==2.9.0

gdown==3.13.0

torchvision==0.8.2

torch==1.7.1

tqdm==4.62.2

backpack==0.1

parameterized==0.8.1

Pillow==8.3.2

Second, please download the datasets:

python3 -m domainbed.scripts.download --data_dir=./domainbed/data/path

Or just download the dataset directly from this link \url{https://drive.google.com/file/d/1uImltLI1oJzxK9paZOt2alQut6wvKqTI/view?usp=sharing} and put it in "./domainbed/data/path".

Third, launch a sweep using:

python -m sweep --command launch --data_dir=./domainbed/data/path --output_dir=./domainbed/outputThuan/path --command_launcher local --algorithms name_of_algorithm --datasets ColoredMNIST --n_hparams 20 --n_trials 3

-> n_trials: number of trials i.e, number of random seeds, for each trial, the algorithm will scan over all the hyper-parameters. --n_hparams: number of hyper-parameter pairs which is randomly picked in a range of (0.1, 10000). --datasets: default by ColoredMNIST, --algorithms: please select one algorithm from 4 algorithms at the end of the algorithm.py.

Fourth, after all jobs have either succeeded or failed, you can delete the data from failed jobs with:

python -m sweep --command delete_incomplete --data_dir=./domainbed/data/path --output_dir=./domainbed/outputThuan/path --command_launcher local --algorithms name_of_algorithm --datasets ColoredMNIST --n_hparams 20 --n_trials 3

and then re-launch them again.

Finally, to view the results of your sweep by:

python -m domainbed.scripts.collect_results --input_dir=/my/sweep/output/path

Note: please take a look at the tested algorithms in "algorithms.py". The algorithms are denoted from #1 to #4 in order: MMD + reconstruction loss, IRM + reconstruction loss, CORAL + reconstruction loss, and ERM + reconstruction loss. Please select only one algorithm from these 04 algorithms to run at a time and comment out other ones. In the default code, we are using the "MMD + reconstruction loss algorithm" and commented out other ones.

Our code is based on this repo at https://github.com/facebookresearch/DomainBed.

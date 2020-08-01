# ai602_fl
Hello World

## Environments
```shell script
python == 3.7.0
pytorch == 1.5.0
yaml
seaborn
matplotlib
pandas
```
Please install required packages through ```pip install -r requirements.txt```

## Results
All the results related with experiments will be saved in the log folder.
You might not need to create an empty ```log``` folder.

## Run
Related hyperparameters are in config/config.yaml. Configuration can be controlled config/config.yaml.
Run with the following script: * Caution on UnicodeDecodeError for config.yaml
```python main.py```

If you want to experiments several settings, please modify ```config.yaml``` or argparse.  
```shell script
usage: main.py [-h] [--config_file CONFIG_FILE] [--nb_exp_reps NB_EXP_REPS]
               [--nb_devices NB_DEVICES] [--nb_rounds NB_ROUNDS] [--lr LR]
               [--model MODEL] [--pruning_type PRUNING_TYPE]
               [--plan_type PLAN_TYPE] [--decay_type DECAY_TYPE]
               [--device DEVICE] [--scheduler SCHEDULER]
               [--target_sparsity TARGET_SPARSITY] [--local_ep LOCAL_EP]
               [--cuda_type CUDA_TYPE] [--weight_decay WEIGHT_DECAY]
               [--dataset DATASET]
```

## Plot
You can plot with using ```plot_figure.py```. 
```shell script
usage: plot_figure.py [-h] [--root ROOT] [--xs XS [XS ...]]
                      [--exp_name EXP_NAME [EXP_NAME ...]]
                      [--legend LEGEND [LEGEND ...]] [--title TITLE]
```
**Check the example below**
- root: the root location of your experiments. Defaults is './log/'
- xs: choose x-axis. [str, list] list can be parsed with the white space.
- exp_name: write down all exp_name which you would like to compare. [str, list]
- legend: if you want to set the label for each exp, use this. The order should be exactly same wwith exp_name. [str, list]
- title: the title for saving figure.
 
Example:
```shell script
python plot_figure.py  
       --xs round  
       --exp_name [mlp-mnist]vanilla_lr_step_0.03_localep_1 [deep_mlp-mnist]vanilla_lr_step_0.03_localep_1 [mlp-mnist]vanilla_lr_step_0.03_localep_5 [deep_mlp-mnist]vanilla_lr_step_0.03_localep_5
       --legend mlp_local_ep1 deep_local_ep1 mlp_local_ep5 deep_local_ep5
       --title mlp_vs_deep
```
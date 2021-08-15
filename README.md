# FEWER: Federated Weight Recovery

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
Run with the following script:

```python main.py```

If you want to experiments several settings, you need to make each configuration yaml file in the 'config/settings' folder.
Then You can run all the experiments with the following script:

```python main.py setings```

* Caution on UnicodeDecodeError for config.yaml

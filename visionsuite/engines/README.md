# Tutorial

## Directory Structure
```
<task>
   |----- cfgs
            |----- dataset
                     |----- xxx.yaml
            |----- default.yaml
   |----- src
           |----- dataloaders
                     |----- root.py
                     |----- xxx.py
           |----- datasets
                     |----- root.py
                     |----- xxx.py
           |----- loops
                     |----- root.py
                     |----- xxx.py
           |----- losses
           |----- models
           |----- optimizers         
           |----- runners
           |----- schedulers
           |----- trainers
           |----- validators
   |----- utils
           |----- augment
           |----- metrics
           |----- results
           |----- vis
```

## Naming Rules
- File names: ***snake_case***
- Class names: ***CamelCase***
- File-Class alignment: Each file name should correspond to the class name it contains, with the file name reflecting the class in snake_case
    - For example, if the class name is CustomDataset, then the file should be named custom_dataset.py.


## Add new module
1. OOP is preferable
2. Register the module with decorator
3. Importing in root.py and including it in the __all__ list.
 


This program is made to facilitate the Digital Forensics project of the Huygens KNAW.


use python3 train_siamese_network -h for help on general parameters that are not documented here.


The siamese network contains several predefined models (to be added in this documentation later) and the option the create simple network using "spec"

# Spec
Spec is inspired by the VGSL-spec, but it might be slightly different here and there.

## Convolution layer
C: Convolution2D  

l: leakyRelu  
t: tanh  
r: relu  
s: sigmoid  
e: elu

example layer:
Cl3,3,8  : Convolution2D with 3 by 3 kernel and 8 outputs  
please note specification of the activation neuron type is needed always use `l`, `t` , `r` , `s` or `e` after `C`

## Fully connected (dense) layer
F: Dense

example:
F64

## MaxPool layer
Mp: MaxPool

example layer:
Mp2  : MaxPool 2 x 2

## Global max pooling & global average pooling
Gm : global max pooling  
Ga : global average pooling

All layers should be separated with the space character

## Dropout
D: Dropout  
example: D0.10

## Full example
example full spec:

Cl3,3,8 Mp2,2 Cl3,3,32 Gm F96

you can call the program like this:
```
python3 train_siamese_network --spec "Cl3,3,8 Mp2,2 Cl3,3,32 Gm F96"
```

## Quirks
- The software expects a final fully connected layer with a maximum of 96 outputs (F96).

## Export results with stored weights

```
python3.8 export_siamese_network_weights.py --weights_path /path/to/checkpoint --dir /path/to/folder_with_images --results_file ./results.csv
```
`/path/to/checkpoint` will contain the following files: 
* `checkpoint`
* `difornet40_weights.data-00000-of-00001`
* `difornet40_weights.index`

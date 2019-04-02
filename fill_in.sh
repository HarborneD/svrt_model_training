#!/bin/sh

num_train_steps=80
for learning_rate in 0.00001 0.000001 0.0001
do
    for model in "inception_v3_imagenet" #"vgg16" "vgg16_imagenet" "vgg19_imagenet"
    do
        for train_size in 1000 2000
        do
            for i in $(seq 1 23)
            do
                python train_svrt_model.py "SVRT Problem $i" $model $train_size $num_train_steps $learning_rate
            
            done
        done
    done
done

#! /usr/bin/env bash

for i in pong qbert spaceinvaders videopinball; do 
    echo "Running behavioral cloning on $i"
    python main.py --env_name $i
done

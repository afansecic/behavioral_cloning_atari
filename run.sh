#!/bin/bash
OUTPUT_FILE=/u/prabhatn/behavioral_cloning_atari/checkpoints/pong/stdout.txt
ROM=/u/prabhatn/ale/thesis/dqn/roms/pong.bin
CHECKPOINT_DIR=/u/prabhatn/behavioral_cloning_atari/checkpoints/pong
python main.py --rom $ROM \
--updates 500000 \
--checkpoint-dir $CHECKPOINT_DIR &>> $OUTPUT_FILE

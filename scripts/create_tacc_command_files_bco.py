def create_command_file(envname):
    command_str = "module load tacc-singularity\n"
    command_str += 'singularity exec --nv ${SINGULARITY_CACHEDIR}/tacc-maverick-ml-latest.simg python -c "import torch; print(torch.cuda.is_available())"\n'
    command_str += "singularity exec --nv ${SINGULARITY_CACHEDIR}/tacc-maverick-ml-latest.simg python main_bco.py --env_name " + envname + " --num_transitions 20000 --use_best 0.25 --checkpoint-dir checkpoints_bco_best\n"
    print(command_str)
    f = open("commands_best_" + envname,'w')
    f.write(command_str)
    f.close()
    
envs = ['mspacman', 'videopinball', 'hero', 'beamrider', 'qbert', 'seaquest', 'breakout', 'spaceinvaders', 'pong', 'enduro' ]
for e in envs:
    print("+"*20)
    print(e)
    print("+"*20)
    create_command_file(e)

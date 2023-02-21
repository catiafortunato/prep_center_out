# !/bin/bash
# Script to run simulations
export seed_str='0 1 2 3 4 5 6 7 8 9'
export seed_str='0'
export sds=($seed_str)
export nseeds=${#sds[@]}
export threads=1
export incr=$((nseeds / threads))

for ((i=0;i<$threads;i++)); 
do
    export i=$i
    screen -dmS run$i bash -c \
    'seeds=($seed_str)
    for seed in ${seeds[@]:$((i*incr)):$(((i+1)*incr))}
    do  
        # #simulate standard networks
        # python ./run_pipeline.py 100006$seed 1 -c standard_config.yaml -file Chewie_CO_CS_2016-10-14

        #simulate decorrelated networks
        python ./run_pipeline.py 100006$seed 1 -c decorrelated_config.yaml -file Chewie_CO_CS_2016-10-14
    done'
done



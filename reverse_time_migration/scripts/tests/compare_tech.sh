#!/bin/bash

./config.sh -t omp -i on
./clean_build.sh
./bin/Engine -w ./results_omp

./config.sh -t dpc -i on
./clean_build.sh
./bin/Engine -w ./results_dpc_cpu -p ./data/computation_parameters_dpc_cpu.json
./bin/Engine -w ./results_dpc_gpu -p ./data/computation_parameters_dpc_gen9.json

cd bin/
make compare_csv
cd ../

step=200
max_nt=14400
mkdir comparison_results
mkdir comparison_results/comp_forward
mkdir comparison_results/comp_reverse
mkdir comparison_results/comp_backward
mkdir comparison_results/comp_vel
mkdir comparison_results/comp_trace
mkdir comparison_results/comp_mig
for j in results_omp results_dpc_cpu results_dpc_gpu; do
  found_match=0
  for k in results_omp results_dpc_cpu results_dpc_gpu; do
    if [ $k != $j ] && [ $found_match == 1 ]; then
      ./bin/utils/compare_csv ./$k/csv/velocity.csv ./$j/csv/velocity.csv >comparison_results/comp_vel/$j-$k-vel.txt
      ./bin/utils/compare_csv ./$k/csv/traces/trace_0.csv ./$j/csv/traces/trace_0.csv >comparison_results/comp_trace/$j-$k-trace.txt
      ./bin/utils/compare_csv ./$k/csv/migration.csv ./$j/csv/migration.csv >comparison_results/comp_mig/$j-$k-mig.txt
      for i in forward backward reverse; do
        ./bin/utils/compare_csv $step $max_nt ./$k/csv/$i/ ./$j/csv/$i/ $i >comparison_results/comp_$i/$j-$k-$i.txt
      done
    elif [ $k == $j ]; then
      found_match=1
    fi
  done
done

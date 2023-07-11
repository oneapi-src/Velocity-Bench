#!/bin/bash

#to run the script give it the number of cores
if [[ $# -eq 0 ]]; then
  echo 'To use this script you will need to give it 1 argument...'
  echo 'The argument should specify the maximum number of cores/threads to create'
  exit 0
fi
rm -f ../results/full_time_results.txt
touch ../results/full_time_results.txt
echo "threads,\treal,\tuser,\tsystem" >../results/full_time_results.txt
for ((i = 1; i <= $1; i++)); do
  echo "Running with  $i threads"
  sed -i "s/thread-number=[0-9]\+/thread-number=$i/" ../data/computation_parameters.txt
  /usr/bin/time -ao ../results/full_time_results.txt -f "$i,\t%E,\t%U,\t%S" ../bin/acoustic_engine
done

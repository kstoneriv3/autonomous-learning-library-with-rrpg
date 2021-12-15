#!/bin/bash

vpg(){
  python scripts/classic.py CartPole-v1 vpg --frames 200000 --quiet True --device cpu
}

rrpg(){
  python scripts/classic.py CartPole-v1 rrpg --frames 650000  --quiet True --device cpu
}

qmcpg(){
  python scripts/classic_qmc.py --frames 200000 --quiet True --device cpu
}

all_tasks(){
  source $venv/bin/activate
  echo "[$(date)]: Starting VPG ..."
  vpg
  echo "[$(date)]: Starting RRPG ..."
  rrpg
  echo "[$(date)]: Starting QMCPG..."
  qmcpg
  echo "[$(date)]: Finished all three tasks."
}

open_sem(){
  mkfifo pipe-$$
  exec 3<>pipe-$$
  rm pipe-$$
  local i=$1
  for((;i>0;i--)); do
    printf %s 000 >&3
  done
}

# run the given command asynchronously and pop/push tokens
run_with_lock(){
  local x
  # this read waits until there is something to read
  read -u 3 -n 3 x && ((0==x)) || exit $x
  (
    ( "$@"; )
  # push the return code of the command to the semaphore
    printf '%.3d' $? >&3
  )&
}

read -p 'The number of iterations: ' n_iter
read -p 'The path of the virtual env: ' venv
venv=${venv:-~/Desktop/dev_env_rl}

N=6
open_sem $N
for ((j=1;j<=$n_iter;j++)); do
  #run_with_lock rrpg
  #run_with_lock qmcpg
  #run_with_lock vpg
  run_with_lock all_tasks
done 


#!/bin/sh
# All runs of the Li+/Mg2+/Br- extension (two parallel chains, one CPU thread each).
cd "$(dirname "$0")"; mkdir -p ../results/s4_logs
export S2_TAG=s2v2 TORCH_THREADS=1
r() { python s4_extend.py "$@" >> ../results/s4_logs/s4_$1_$2_$3$4.log 2>&1; }
( r full 0 all; r scratch 0 all ) &          # (already run; kept here for completeness when rerunning from scratch)
wait
( r frozen 0 all; r frozen 0 300; r frozen 0 1000; r frozen 0 3000; r frozen 0 10000
  r frozen 0 all release; r frozen 1 all release; r frozen 2 all release ) &
( r frozen 1 all; r full 1 all; r frozen 3 all release; r frozen 4 all release ) &
wait

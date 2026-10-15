#!/usr/bin/env bash
# nppdx-wins W2, full protocol (100 warm-up + 1000 timed iterations, every mode), default clocks.
# The smoke's choices are re-swept, except where the smoke's choice beat the runner-up by more than 2% (its noise):
# only fkl_staged's DPP tile 32x16 for the 5x5 box in two passes (6.1% ahead), pinned with WINS_PINNED_CHOICES.
set -e
B=/home/oscar/bvf-bench/builds/wins-full
D=/home/oscar/bvf-bench/results-nppdx-wins-full
cd "$D"
git -C /home/oscar/bvf-bench/paper-bench-wins rev-parse HEAD > commit.txt
git -C /home/oscar/bvf-bench/paper-bench-wins/fkl rev-parse HEAD >> commit.txt
for c in k5 k5_p2 k9 gauss21; do
    target=benchmark_wins_frame_$c
    [ "$c" = k5 ] && target=benchmark_wins_frame
    if [ "$c" = k5_p2 ]; then
        WINS_PINNED_CHOICES="$D/pinned_k5_p2.csv" "$B/$target" "$D/$c" > "$D/${c}_main.log" 2>&1
    else
        "$B/$target" "$D/$c" > "$D/${c}_main.log" 2>&1
    fi
    "$B/$target" "$D/$c" --gpu-only-l2-flushed > "$D/${c}_flushed.log" 2>&1
    echo "$c done $(date +%T)"
done

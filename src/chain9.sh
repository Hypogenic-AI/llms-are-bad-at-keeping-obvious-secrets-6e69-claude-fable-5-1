#!/bin/bash
# Final GPU job chain (replaces chain4-8, which waited on a zombie pid and never started)
set -x
E1=../results/exp1/trials; E2=../results/exp2/trials; E3=../results/exp3/trials
python exp1b_generate.py > ../logs/exp1b_generate.log 2>&1
python exp1c_generate.py > ../logs/exp1c_generate.log 2>&1
python build_trials.py exp1
python local_judge.py g12 $E1/disc_filler_craft_long.jsonl $E1/disc_filler_irrelevant_plain.jsonl > ../logs/judge_exp1b.log 2>&1
echo STEP1_DONE
python exp2b_generate.py > ../logs/exp2b_generate.log 2>&1
python build_trials.py exp2
python local_judge.py g12 $E2/disc_open_base.jsonl $E2/disc_open_plan_blind.jsonl > ../logs/judge_exp2b.log 2>&1
echo STEP2_DONE
python exp3_swap.py 6 > ../logs/exp3_swap.log 2>&1
python build_trials.py exp3
python local_judge.py g12 $E3/disc_swap_early30.jsonl $E3/disc_swap_early100.jsonl $E3/disc_swap_late30.jsonl $E3/disc_swap_late100.jsonl > ../logs/judge_exp3_swap.log 2>&1
echo STEP3_DONE
python local_judge.py g12 $E1/disc_decoy_asdecoy.jsonl > ../logs/judge_exp1c.log 2>&1
python exp3_extract.py filler_craft_long filler_irrelevant_plain > ../logs/exp3_extract_b.log 2>&1
echo STEP4_DONE
python subsample_trials.py 48 ../results/exp2/trials_g27 $E2/disc_base.jsonl $E2/disc_plan_blind.jsonl $E2/disc_open_base.jsonl $E2/disc_open_plan_blind.jsonl
python subsample_trials.py 48 ../results/exp1/trials_g27 $E1/disc_dont_reveal.jsonl $E1/disc_plan_blind.jsonl
G=../results/exp2/trials_g27; H=../results/exp1/trials_g27
python local_judge.py g27 $G/disc_base.jsonl $G/disc_plan_blind.jsonl $G/disc_open_base.jsonl $G/disc_open_plan_blind.jsonl $H/disc_dont_reveal.jsonl $H/disc_plan_blind.jsonl > ../logs/judge_g27.log 2>&1
echo CHAIN9_DONE

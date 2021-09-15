#! /bin/bash
# Run conformal evaluation on IR for fact verification.

set -ex

# Need split!
if [ "$#" -ne 1 ]
then
  echo "Please supply split."
  exit 1
fi

# Take the split as a commandline arg.
split=$1

# Compute baseline CP without expanded admission.
python ir/conformal_ir.py \
       --eval_file data/ir/preds_$split.jsonl \
       --gold_file data/ir/$split.jsonl \
       --metrics=logit \
       --num_trials=50 \
       --outer_threads=50 \
       --inner_threads=0 \
       --cuda=false \
       --skip_conformal=false \
       --skip_baselines=true \
       --equivalence=false \
       --output_dir=results/$split \
       --smoothed=true \
       --absolute_efficiency \
       --epsilons=0.01,0.05,0.1,0.2

# Compute noncascade CP with expanded admission.
python ir/conformal_ir.py \
       --eval_file data/ir/preds_$split.jsonl \
       --gold_file data/ir/$split.jsonl \
       --metrics=logit \
       --num_trials=50 \
       --outer_threads=50 \
       --inner_threads=0 \
       --cuda=false \
       --skip_conformal=false \
       --skip_baselines=true \
       --equivalence=true \
       --output_dir=results/$split \
       --smoothed=true \
       --absolute_efficiency \
       --epsilons=0.01,0.05,0.1,0.2


# Compute cascaded CP with expanded admission.
python ir/conformal_ir.py \
       --eval_file data/ir/preds_$split.jsonl \
       --gold_file data/ir/$split.jsonl \
       --correction=simes \
       --metrics=bm25,logit \
       --num_trials=50 \
       --outer_threads=50 \
       --inner_threads=0 \
       --cuda=false \
       --skip_conformal=false \
       --skip_baselines=true \
       --equivalence=true \
       --output_dir=results/$split \
       --smoothed=true \
       --absolute_efficiency \
       --epsilons=0.01,0.05,0.1,0.2

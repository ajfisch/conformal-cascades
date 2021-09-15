#! /bin/bash
# Run conformal evaluation on ChEMBL combinatorial property constraint task.

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
python chembl/conformal_chembl.py \
       --metrics=MPN \
       --eval_dir data/chembl/$split \
       --num_trials 10 \
       --outer_threads 0 \
       --inner_threads 40 \
       --cuda=true \
       --skip_conformal=false \
       --skip_baselines=true \
       --equivalence=false \
       --output_dir=results/$split \
       --smoothed=true \
       --absolute_efficiency \
       --epsilons=0.1,0.2,0.3,0.4

# Compute noncascade CP with expanded admission.
python chembl/conformal_chembl.py \
       --metrics=MPN \
       --eval_dir data/chembl/$split \
       --num_trials 10 \
       --outer_threads 0 \
       --inner_threads 40 \
       --cuda=true \
       --skip_conformal=false \
       --skip_baselines=true \
       --equivalence=true \
       --output_dir=results/$split \
       --smoothed=true \
       --absolute_efficiency \
       --epsilons=0.1,0.2,0.3,0.4

# Compute cascaded CP with expanded admission.
python chembl/conformal_chembl.py \
       --metrics=RF,MPN \
       --correction=bonferroni \
       --eval_dir data/chembl/$split \
       --num_trials 10 \
       --outer_threads 0 \
       --inner_threads 40 \
       --cuda=true \
       --skip_conformal=false \
       --skip_baselines=true \
       --equivalence=true \
       --output_dir=results/$split \
       --smoothed=true \
       --absolute_efficiency \
       --epsilons=0.1,0.2,0.3,0.4

#! /bin/bash
# Run conformal evaluation on open-domain qa.

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
python qa/conformal_open_qa.py \
       --metrics=sum \
       --smoothed=true \
       --eval_file=data/open-qa/preds_$split.json \
       --cache_data=true \
       --num_trials=20 \
       --outer_threads=0 \
       --inner_threads=40 \
       --cuda=true \
       --skip_baselines=true \
       --equivalence=false \
       --output_dir=results/$split \
       --absolute_efficiency \
       --epsilons=0.1,0.2,0.3,0.4

# Compute noncascade CP with expanded admission.
python qa/conformal_open_qa.py \
       --metrics=sum \
       --smoothed=true \
       --eval_file=data/open-qa/preds_$split.json \
       --cache_data=true \
       --num_trials=20 \
       --outer_threads=0 \
       --inner_threads=40 \
       --cuda=true \
       --skip_baselines=true \
       --equivalence=true \
       --output_dir=results/$split \
       --absolute_efficiency \
       --epsilons=0.1,0.2,0.3,0.4

# Compute cascaded CP with expanded admission.
python qa/conformal_open_qa.py \
       --metrics=psg_score,relevance_logit,start_logit,end_logit \
       --smoothed=true \
       --correction=simes \
       --eval_file=data/open-qa/preds_$split.json \
       --cache_data=true \
       --num_trials=20 \
       --outer_threads=0 \
       --inner_threads=40 \
       --cuda=true \
       --skip_baselines=true \
       --equivalence=true \
       --output_dir=results/$split \
       --absolute_efficiency \
       --epsilons=0.1,0.2,0.3,0.4

# Drug Therapeutics Program (DTP) AIDS Antiviral Screen Task

This is a proof-of-concept of in-silico screening for drug repurposing/discovery. Data and models are available through [chemprop](https://github.com/chemprop/chemprop).

### Chemprop Models

Train and evaluate models using:
```
./run_chemprop.sh
```

Then aggregate results:
```
python aggregate.py
```

### Conformal Prediction
- Run `conformal_hiv.py`.

"""Prepare RDKit features for chemprop models."""

import argparse
import glob
import multiprocessing
import os
import tqdm
import numpy as np
import chemprop


def init_fn(features_generator):
    global generator
    generator = features_generator


def generate_features(smiles):
    global generator
    features = np.array(generator(smiles))
    features = np.where(np.isnan(features), 0, features)
    return smiles, features


def generate_features_for_file(smiles_path, features_path, features_generator, num_workers):
    smiles_to_idx = {}
    idx_to_features = []
    features_generator = chemprop.features.get_features_generator(features_generator)
    workers = multiprocessing.Pool(num_workers, initializer=init_fn, initargs=(features_generator,))
    all_smiles = list(chemprop.data.get_smiles(smiles_path))
    with tqdm.tqdm(total=len(all_smiles), desc="generating features") as pbar:
        for i, (smiles, features) in enumerate(workers.imap_unordered(generate_features, all_smiles)):
            smiles_to_idx[smiles] = i
            idx_to_features.append(features)
            pbar.update()
    idx_to_features = np.stack(idx_to_features)
    np.save(features_path, (smiles_to_idx, idx_to_features))


def main(args):
    smiles_paths = glob.glob(args.file_pattern)
    features_paths = []
    for smiles_file in smiles_paths:
        basename = os.path.splitext(os.path.basename(smiles_file))[0]
        features_dir = args.features_dir
        if features_dir is None:
            features_dir = os.path.dirname(smiles_file)
        os.makedirs(features_dir, exist_ok=True)
        features_paths.append(os.path.join(features_dir, basename + ".npy"))
    for smiles_path, features_path in zip(smiles_paths, features_paths):
        generate_features_for_file(smiles_path, features_path, args.features_generator, args.num_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=multiprocessing.cpu_count() - 5)
    parser.add_argument("--file_pattern", type=str, default="../data/chembl/*.csv")
    parser.add_argument("--features_dir", type=str, default="../data/chembl/features")
    parser.add_argument("--features_generator", type=str, default="rdkit_2d_normalized")
    main(parser.parse_args())

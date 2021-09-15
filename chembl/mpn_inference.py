"""Inference with ensembles."""

import argparse
import os
import tqdm
import torch
import mpn
import json


def main(args):
    dataset = mpn.IndependentMPN.load_dataset(
        dataset_file=args.eval_data,
        features_file=args.eval_features,
        features_generator=args.features_generator)

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_data_workers,
        collate_fn=mpn.construct_molecule_batch)

    smiles_to_logits = {}
    num_models = len(args.checkpoints)
    for path in tqdm.tqdm(args.checkpoints, desc="evaluating models"):
        model = mpn.IndependentMPN.load_from_checkpoint(path).cuda()
        model.eval()
        for inputs, batch_smiles in tqdm.tqdm(loader, desc="predicting"):
            with torch.no_grad():
                inputs = model.transfer_batch_to_device(inputs, model.device)
                outputs = model.forward(inputs)
                for i, smiles in enumerate(batch_smiles):
                    if smiles not in smiles_to_logits:
                        smiles_to_logits[smiles] = outputs["logits"][i].cpu().detach() / num_models
                    else:
                        smiles_to_logits[smiles] += outputs["logits"][i].cpu().detach() / num_models

    smiles_to_logits = {k: torch.sigmoid(v).tolist() for k, v in smiles_to_logits.items()}
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(smiles_to_logits, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="../ckpts/chembl/independent_mpn/pred_test_ensemble.json")
    parser.add_argument("--eval_data", type=str, default="../data/chembl/test_molecules.csv")
    parser.add_argument("--eval_features", type=str, default="../data/chembl/features/test_molecules.npy")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None)
    parser.add_argument("--features_generator", type=str, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_data_workers", type=int, default=20)
    args = parser.parse_args()
    main(args)

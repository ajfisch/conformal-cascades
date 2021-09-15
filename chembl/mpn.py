"""Independent MPN pre-training."""

import argparse
import logging
import shutil
import subprocess
import tqdm
import os
import csv
import chemprop
import encoders
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def construct_molecule_batch(batch):
    smiles = [x.smiles for x in batch]
    inputs = chemprop.data.data.construct_molecule_batch(batch)
    mol_graph = inputs.batch_graph()
    if mol_graph:
        mol_graph = mol_graph.get_components()
    mol_features = inputs.features()
    if mol_features:
        mol_features = torch.from_numpy(np.stack(mol_features)).float()
    targets = inputs.targets()
    if targets:
        mask = [[1 if t is not None else 0 for t in ex] for ex in targets]
        targets = [[t if t is not None else 0 for t in ex] for ex in targets]
        targets = torch.Tensor(targets).float()
        mask = torch.Tensor(mask).float()
    inputs = [mol_graph, mol_features, targets, mask]
    return inputs, smiles


class IndependentMPN(pl.LightningModule):

    def __init__(self, hparams):
        super(IndependentMPN, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.encoder = encoders.MoleculeEncoder(hparams)
        self.classifier = nn.Linear(hparams.enc_hidden_size, hparams.num_targets)

    def forward(self, inputs):
        mol_graph, mol_features, targets, mask = inputs
        mol_encs = self.encoder(mol_graph, mol_features)
        logits = self.classifier(mol_encs)
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(logits, targets, weight=mask, reduction="none")
            loss = loss.sum() / mask.sum()
        else:
            loss = None
        return dict(logits=logits, probs=torch.sigmoid(logits), loss=loss, targets=targets, mask=mask)

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        outputs = self.forward(inputs)
        loss = outputs["loss"]
        tensorboard_logs = {"train_loss": loss.detach()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, _ = batch
        return self.forward(inputs)

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["probs"].detach() for x in outputs], dim=0).tolist()
        targets = torch.cat([x["targets"].detach() for x in outputs], dim=0).tolist()
        mask = torch.cat([x["mask"].detach() for x in outputs], dim=0).tolist()
        for i, example in enumerate(mask):
            for j, target in enumerate(example):
                if not target:
                    targets[i][j] = None
        results = {}
        for metric in ["auc", "cross_entropy"]:
            values = chemprop.train.evaluate_predictions(
                preds=preds,
                targets=targets,
                num_tasks=self.hparams.num_targets,
                metric_func=chemprop.utils.get_metric_func(metric),
                dataset_type="classification",
                logger=logging.getLogger(__name__))
            results[metric] = torch.tensor(np.mean([v for v in values if not np.isnan(v)]))
        tensorboard_logs = results
        tqdm_dict = tensorboard_logs
        return {"val_loss": results["cross_entropy"], "progress_bar": tqdm_dict, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-6)
        return [optimizer], [scheduler]

    @staticmethod
    def load_dataset(dataset_file, features_file=None, features_generator=None, limit=None):
        if features_file is not None:
            smiles_to_idx, idx_to_features = np.load(features_file, allow_pickle=True)
        mols = []
        num_lines = int(subprocess.check_output(["wc", "-l", dataset_file], encoding="utf8").split()[0])
        with open(dataset_file, "r") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            for row in tqdm.tqdm(reader, total=num_lines - 1, desc="reading dataset"):
                smiles = row[columns[0]]
                targets = [float(row[c]) if row[c] else None for c in columns[1:]]
                mol = chemprop.data.MoleculeDatapoint(
                    smiles=smiles,
                    targets=targets,
                    features=idx_to_features[smiles_to_idx[smiles]] if features_file else None,
                    features_generator=features_generator)
                mols.append(mol)

        if limit:
            idx = np.random.default_rng(0).permutation(len(mols))[:limit]
            mols = [mols[i] for i in idx]

        dataset = chemprop.data.MoleculeDataset(mols)
        return dataset

    def prepare_data(self):
        for split in ["train", "val"]:
            dataset = self.load_dataset(
                dataset_file=getattr(self.hparams, "%s_data" % split),
                features_file=getattr(self.hparams, "%s_features" % split),
                features_generator=self.hparams.features_generator,
                limit=None if split == "train" else self.hparams.subsample_val)
            setattr(self, "%s_dataset" % split, dataset)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_molecule_batch)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_molecule_batch)
        return loader

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.register("type", "bool", pl.utilities.parsing.str_to_bool)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--gpus", type=int, nargs="+", default=None)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--max_epochs", type=int, default=40)
        parser.add_argument("--checkpoint_dir", type=str, default="../ckpts/chembl/independent_mpn")
        parser.add_argument("--overwrite", type=bool, default=True)

        parser.add_argument("--num_data_workers", type=int, default=20)
        parser.add_argument("--train_data", type=str, default="../data/chembl/train_molecules.csv")
        parser.add_argument("--train_features", type=str, default="../data/chembl/features/train_molecules.npy")
        parser.add_argument("--val_data", type=str, default="../data/chembl/val_molecules.csv")
        parser.add_argument("--val_features", type=str, default="../data/chembl/features/val_molecules.npy")
        parser.add_argument("--subsample_val", type=int, default=10000)

        parser.add_argument("--num_targets", type=int, default=1227)
        parser.add_argument("--features_generator", type=str, nargs="+", default=None)
        parser.add_argument("--use_mpn_features", type="bool", default=True)
        parser.add_argument("--use_mol_features", type="bool", default=True)
        parser.add_argument("--mpn_hidden_size", type=int, default=1024)
        parser.add_argument("--mol_features_size", type=int, default=200)
        parser.add_argument("--ffnn_hidden_size", type=int, default=1024)
        parser.add_argument("--num_ffnn_layers", type=int, default=3)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--mpn_depth", type=int, default=3)
        parser.add_argument("--undirected_mpn", type="bool", default=False)
        parser.add_argument("--enc_hidden_size", type=int, default=1024)

        return parser


def main(args):
    pl.seed_everything(args.seed)
    model = IndependentMPN(hparams=args)
    print(model)
    if os.path.exists(args.checkpoint_dir) and os.listdir(args.checkpoint_dir):
        if not args.overwrite:
            raise RuntimeError("Experiment directory is not empty.")
        else:
            shutil.rmtree(args.checkpoint_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.checkpoint_dir, "weights.pt"),
        save_top_k=1,
        verbose=True,
        monitor="auc",
        mode="max")
    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=args.checkpoint_dir,
        version="logs")
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        terminate_on_nan=True)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = IndependentMPN.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)

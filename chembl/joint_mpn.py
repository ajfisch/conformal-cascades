"""Joint MPN prediction.

f: ({target_1, ..., target_k}, molecule) ---> {0, 1}
"""

import argparse
import chemprop
import encoders
import json
import numpy as np
import os
import pytorch_lightning as pl
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm


class CombinationDatapoint(chemprop.data.MoleculeDatapoint):

    def __init__(self, smiles, combination, assignment, label, features=None, features_generator=None):
        super(CombinationDatapoint, self).__init__(
            smiles, features=features, features_generator=features_generator)
        self.combination = combination
        self.assignment = assignment
        self.label = label


class TruncatedSampler:

    def __init__(self, dataset_size, batch_size, max_batches=None):
        self.N = dataset_size
        if max_batches is None:
            self.max_batches = dataset_size // batch_size
        else:
            self.max_batches = min(dataset_size // batch_size, max_batches)
        self.batch_size = batch_size

    def __iter__(self):
        indices = np.random.permutation(self.N)
        batches = [indices[i * self.batch_size:(i + 1) * self.batch_size]
                   for i in range(self.max_batches)]
        return iter(batches)

    def __len__(self):
        return self.max_batches


def construct_molecule_batch(batch):
    smiles = [x.smiles for x in batch]

    # Get constraint setting.
    combination = torch.LongTensor([x.combination for x in batch])
    assignment = torch.FloatTensor([x.assignment for x in batch])
    labels = torch.FloatTensor([x.label for x in batch])

    # Get molecules.
    inputs = chemprop.data.data.construct_molecule_batch(batch)
    mol_graph = inputs.batch_graph()
    if mol_graph:
        mol_graph = mol_graph.get_components()
    mol_features = inputs.features()
    if mol_features:
        mol_features = torch.from_numpy(np.stack(mol_features)).float()

    inputs = [mol_graph, mol_features, combination, assignment, labels]
    return inputs, smiles


class JointMPN(pl.LightningModule):

    def __init__(self, hparams):
        super(JointMPN, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams

        with open(hparams.targets_file, "r") as f:
            self.index2target = json.load(f)
        self.target2index = {t: i for i, t in enumerate(self.index2target)}

        self.encoder = encoders.MoleculeEncoder(hparams)
        self.embedding = nn.Embedding(hparams.num_targets, hparams.enc_hidden_size)
        self.transform = nn.Linear(hparams.enc_hidden_size + 1, hparams.enc_hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hparams.enc_hidden_size, hparams.enc_hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.enc_hidden_size, hparams.enc_hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.enc_hidden_size, 1))

        # Warm start parameters.
        if hparams.warm_start:
            print("Warm starting parameters")
            weights = torch.load(hparams.warm_start)["state_dict"]
            state = self.state_dict()
            state["embedding.weight"] = weights["classifier.weight"]
            for k, v in weights.items():
                if k.startswith("encoder"):
                    state[k] = v

    def forward(self, inputs):
        mol_graph, mol_features, combination, assignment, labels = inputs

        # [batch_size, hidden_size]
        mol_encs = self.encoder(mol_graph, mol_features)

        # [batch_size, num_constraints, hidden_size]
        combination_encs = self.embedding(combination)
        assignment = assignment.unsqueeze(-1)
        combination_encs = torch.cat([combination_encs, assignment], dim=-1)

        # [batch_size, hidden_size]
        constraint_enc = F.relu(self.transform(combination_encs)).sum(dim=1)

        # [batch_size, 1]
        combined = torch.cat([constraint_enc, mol_encs], dim=-1)
        logits = self.classifier(combined).squeeze(-1)

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = None
        return dict(logits=logits, probs=torch.sigmoid(logits), loss=loss)

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
        loss = torch.stack([x["loss"].detach() for x in outputs]).mean()
        tensorboard_logs = {"val_loss": loss}
        tqdm_dict = tensorboard_logs
        return {"val_loss": loss, "progress_bar": tqdm_dict, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-6)
        return [optimizer], [scheduler]

    def load_dataset(self, dataset_file, features_file=None, features_generator=None, limit=None):
        if features_file is not None:
            smiles_to_idx, idx_to_features = np.load(features_file, allow_pickle=True)

        mols = []
        num_lines = int(subprocess.check_output(["wc", "-l", dataset_file], encoding="utf8").split()[0])
        with open(dataset_file, "r") as f:
            for line in tqdm.tqdm(f, total=num_lines, desc="reading dataset"):
                data = json.loads(line)
                combination = [self.target2index[t] for t in data["combination"]]
                assignment = data["assignment"]

                # Subsample up to N positives and N negatives.
                negatives = data["negatives"]
                if len(negatives) > self.hparams.subsample_negatives:
                    negatives = np.random.choice(negatives, self.hparams.subsample_negatives)
                for smiles in negatives:
                    mol = CombinationDatapoint(
                        smiles=smiles,
                        combination=combination,
                        assignment=assignment,
                        label=0,
                        features=idx_to_features[smiles_to_idx[smiles]] if features_file else None,
                        features_generator=features_generator)
                    mols.append(mol)

                positives = data["positives"]
                if len(positives) > self.hparams.subsample_positives:
                    positives = np.random.choice(positives, self.hparams.subsample_positives)
                for smiles in positives:
                    mol = CombinationDatapoint(
                        smiles=smiles,
                        combination=combination,
                        assignment=assignment,
                        label=1,
                        features=idx_to_features[smiles_to_idx[smiles]] if features_file else None,
                        features_generator=features_generator)
                    mols.append(mol)

        if limit:
            idx = np.random.permutation(len(mols))[:limit]
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
            batch_sampler=TruncatedSampler(
                dataset_size=len(self.train_dataset),
                batch_size=self.hparams.batch_size,
                max_batches=self.hparams.max_train_batches),
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
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--max_train_batches", type=int, default=3000)
        parser.add_argument("--max_epochs", type=int, default=40)
        parser.add_argument("--checkpoint_dir", type=str, default="../ckpts/chembl/joint_mpn_v4")
        parser.add_argument("--warm_start", type=str, default="../ckpts/chembl/independent_mpn/_ckpt_epoch_6.ckpt")
        parser.add_argument("--overwrite", type=bool, default=True)

        parser.add_argument("--num_data_workers", type=int, default=20)
        parser.add_argument("--targets_file", type=str, default="../data/chembl/targets.json")
        parser.add_argument("--train_data", type=str, default="../data/chembl/train_combinations.jsonl")
        parser.add_argument("--train_features", type=str, default="../data/chembl/features/train_molecules.npy")
        parser.add_argument("--val_data", type=str, default="../data/chembl/val_combinations.jsonl")
        parser.add_argument("--val_features", type=str, default="../data/chembl/features/val_molecules.npy")
        parser.add_argument("--subsample_negatives", type=int, default=50)
        parser.add_argument("--subsample_positives", type=int, default=50)
        parser.add_argument("--subsample_val", type=int, default=50000)

        parser.add_argument("--num_targets", type=int, default=1227)
        parser.add_argument("--features_generator", type=str, nargs="+", default=None)
        parser.add_argument("--use_mpn_features", type="bool", default=True)
        parser.add_argument("--use_mol_features", type="bool", default=True)
        parser.add_argument("--mpn_hidden_size", type=int, default=1024)
        parser.add_argument("--mol_features_size", type=int, default=200)
        parser.add_argument("--ffnn_hidden_size", type=int, default=1024)
        parser.add_argument("--num_ffnn_layers", type=int, default=3)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--mpn_depth", type=int, default=3)
        parser.add_argument("--undirected_mpn", type="bool", default=False)
        parser.add_argument("--enc_hidden_size", type=int, default=1024)

        return parser


def main(args):
    pl.seed_everything(args.seed)
    model = JointMPN(hparams=args)
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
        monitor="val_loss",
        mode="min")
    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=args.checkpoint_dir,
        version="logs")
    trainer = pl.Trainer(
        logger=logger,
        reload_dataloaders_every_epoch=True,
        checkpoint_callback=checkpoint_callback,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        terminate_on_nan=True)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = JointMPN.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)

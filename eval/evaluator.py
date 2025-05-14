import os
import pickle
import torch
import pandas as pd
import numpy as np
import wandb
from tqdm import tqdm
from typing import Dict, List, Tuple
from stats.stats import confidence_interval
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import barrier
from torch.nn.functional import softmax


class Evaluator:
    def __init__(self,
                 model_name_to_model: Dict[str, DDP],
                 gpu_id: int,
                 world_size: int,
                 train_annotations_path: str | None,
                 test_annotations_path: str,
                 train_auc_dataloader: DataLoader | None,
                 test_auc_dataloader: DataLoader | None,
                 slide_scores_dataloader: DataLoader | None,
                 extract_embeddings_dataloader: DataLoader | None,
                 log_train_auc: bool,
                 run_name: str
                 ) -> None:

        self.model_name_to_model = model_name_to_model
        for model_name, model in self.model_name_to_model.items():
            model.to(gpu_id)
        self.gpu_id: int = gpu_id
        self.world_size: int = world_size
        self.train_annotations_path: str = train_annotations_path
        self.train_patient_id_to_slide_name_and_label: dict = dict()
        self.test_patient_id_to_slide_name_and_label: Dict[str, List[Tuple[str, int]]] = dict()
        self.test_annotations_df = pd.read_csv(test_annotations_path)

        self.train_auc_dataloader: DataLoader = train_auc_dataloader
        self.test_auc_dataloader: DataLoader = test_auc_dataloader
        self.slide_scores_dataloader: DataLoader = slide_scores_dataloader
        self.extract_embeddings_dataloader: DataLoader = extract_embeddings_dataloader

        self.run_name: str = run_name

        os.makedirs(f"stats/{self.run_name}", exist_ok=True)

        for annotations_name, annotations_path_and_dict in {
            "train": {"path": train_annotations_path, "dict": self.train_patient_id_to_slide_name_and_label},
            "test": {"path": test_annotations_path, "dict": self.test_patient_id_to_slide_name_and_label},
        }.items():

            if annotations_name == "train" and not log_train_auc:
                continue

            df = pd.read_csv(annotations_path_and_dict["path"])
            for _, row in df.iterrows():
                patient_id = row["patient_id"]
                slide_name = row["slide_name"]
                slide_label = row["slide_label"]
                if patient_id not in annotations_path_and_dict["dict"].keys():
                    annotations_path_and_dict["dict"][patient_id] = [(slide_name, slide_label)]
                else:
                    annotations_path_and_dict["dict"][patient_id].append((slide_name, slide_label))
            del df

    def evaluate_test_auc(self, tested_epoch: int):
        self._auc(
            model_name_to_models=self.model_name_to_model,
            data=self.test_auc_dataloader,
            patient_id_to_slide_name_and_label=self.test_patient_id_to_slide_name_and_label,
            gpu_id=self.gpu_id,
            world_size=self.world_size,
            tested_epoch=tested_epoch,
            dataset_name="test",
            run_name=self.run_name
        )

    def evaluate_train_auc(self, tested_epoch: int):
        self._auc(
            model_name_to_models=self.model_name_to_model,
            data=self.train_auc_dataloader,
            patient_id_to_slide_name_and_label=self.train_patient_id_to_slide_name_and_label,
            gpu_id=self.gpu_id,
            world_size=self.world_size,
            tested_epoch=tested_epoch,
            dataset_name="train",
            run_name=self.run_name
        )

    def evaluate_slide_scores(self):
        df = pd.DataFrame(
            columns=["slide_score", "slide_label", "dataset", "slide_name", "patient_id"] + [str(i) for i in range(500)]
        )
        for _, model in self.model_name_to_model.items():
            model.eval()
        slide_name_to_probabilities = dict()
        device = f"cuda:{self.gpu_id}"

        with torch.no_grad():
            for inputs, labels, slide_name in tqdm(self.slide_scores_dataloader,
                                                   desc=f"Evaluate Slide Scores: GPU [{self.gpu_id}]"):
                list_of_probabilities = []
                for _, model in self.model_name_to_model.items():
                    inputs = inputs.to(device)
                    output = model(inputs)
                    probability = softmax(output[0], dim=0)
                    cpu_probability = probability.cpu().numpy()
                    b_probability = cpu_probability.tolist()[1]
                    list_of_probabilities.append(b_probability)
                mean_probability_on_sample_from_all_models = np.array(list_of_probabilities).mean()
                if slide_name[0] not in slide_name_to_probabilities.keys():
                    slide_name_to_probabilities[slide_name[0]] = [mean_probability_on_sample_from_all_models]
                else:
                    slide_name_to_probabilities[slide_name[0]].append(mean_probability_on_sample_from_all_models)

        if self.gpu_id != 0:
            with open(f"{self.run_name}_outputs_rank_{self.gpu_id}.pkl", "wb") as f:
                pickle.dump(slide_name_to_probabilities, f)

        barrier()

        if self.gpu_id == 0:
            for gpu_id in range(1, self.world_size):
                with open(f"{self.run_name}_outputs_rank_{gpu_id}.pkl", "rb") as f:
                    flushed_slide_name_to_probabilities = pickle.load(f)
                    for slide_name in flushed_slide_name_to_probabilities.keys():
                        if slide_name in slide_name_to_probabilities.keys():
                            slide_name_to_probabilities[slide_name].extend(
                                flushed_slide_name_to_probabilities[slide_name])
                        else:
                            slide_name_to_probabilities[slide_name] = flushed_slide_name_to_probabilities[
                                slide_name]
            for gpu_id in range(1, self.world_size):
                os.remove(f"{self.run_name}_outputs_rank_{gpu_id}.pkl")

            patient_id_to_mean_and_score_dict = dict()

            for patient_id in self.test_patient_id_to_slide_name_and_label.keys():
                patient_slide_names_and_labels = self.test_patient_id_to_slide_name_and_label[patient_id]
                means = []
                for slide_name, label in patient_slide_names_and_labels:
                    probabilities_mean = np.array(
                        slide_name_to_probabilities[slide_name]).mean()
                    means.append(probabilities_mean)
                patient_mean = np.array(means).mean()
                patient_id_to_mean_and_score_dict[patient_id] = {
                    "mean": patient_mean,
                    "label": label
                }
            auc_score, l_interval, r_interval = confidence_interval(patient_id_to_mean_and_score_dict)

            print(
                f"==> AUC for slide scores is: {auc_score}; ({l_interval}, {r_interval})"
            )

            for index, row in tqdm(self.test_annotations_df.iterrows()):
                slide_name = row["slide_name"]
                slide_label = int(row["slide_label"])
                dataset = row["dataset"]
                patient_id = row["patient_id"]
                data = {
                    "slide_score": 0,
                    "slide_label": slide_label,
                    "dataset": dataset,
                    "slide_name": slide_name,
                    "patient_id": patient_id
                }

                probabilities = slide_name_to_probabilities[slide_name]
                for i in range(500):
                    data[str(i)] = [probabilities[i]]
                new_row = pd.DataFrame(data)
                new_row["slide_score"] = new_row.loc[0, [str(i) for i in range(500)]].mean()
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(f"stats/{self.run_name}/slide_scores.xlsx", index=False)

            slide_scores_table = wandb.Table(dataframe=df)
            slide_scores_table_artifact = wandb.Artifact(f"{self.run_name}", type="scores")
            slide_scores_table_artifact.add(slide_scores_table, "slide_scores")
            slide_scores_table_artifact.add_file(f"stats/{self.run_name}/slide_scores.xlsx")
            wandb.log({"slide_scores": slide_scores_table,
                       "AUC": auc_score,
                       "AUC_left_interval": l_interval,
                       "AUC_right_interval": r_interval})
            wandb.log_artifact(slide_scores_table_artifact)

    def extract_embeddings(self):
        #TODO: Fix the multi node problem.
        model_name_to_slide_names_to_embeddings: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = dict()

        with torch.no_grad():
            for inputs, _, slide_name, coordinates in tqdm(self.extract_embeddings_dataloader,
                                                           desc=f"Extract Embeddings: GPU [{self.gpu_id}]"):
                coordinates = coordinates.cpu().detach().numpy()
                for model_name, model in self.model_name_to_model.items():
                    inputs = inputs.to(self.gpu_id)
                    output = model(inputs)
                    output = output.squeeze(-1).squeeze(-1)
                    output = output.cpu().detach().numpy()

                    if model_name not in model_name_to_slide_names_to_embeddings.keys():
                        model_name_to_slide_names_to_embeddings[model_name] = dict()

                    if slide_name[0] not in model_name_to_slide_names_to_embeddings[model_name].keys():
                        model_name_to_slide_names_to_embeddings[model_name][slide_name[0]] = [(output, coordinates)]
                    else:
                        model_name_to_slide_names_to_embeddings[model_name][slide_name[0]] += [(output, coordinates)]

            if self.gpu_id != 0:
                with open(f"embeddings_rank_{self.gpu_id}.pkl", "wb") as f:
                    pickle.dump(model_name_to_slide_names_to_embeddings, f)

        barrier()

        if self.gpu_id == 0:
            for gpu_id in range(1, self.world_size):
                with open(f"embeddings_rank_{gpu_id}.pkl", "rb") as f:
                    flushed_model_name_to_slide_names_to_embeddings = pickle.load(f)
                    for model_name in flushed_model_name_to_slide_names_to_embeddings.keys():
                        if model_name not in model_name_to_slide_names_to_embeddings.keys():
                            model_name_to_slide_names_to_embeddings[model_name] = dict()

                        for slide_name in flushed_model_name_to_slide_names_to_embeddings[model_name].keys():
                            if slide_name not in model_name_to_slide_names_to_embeddings[model_name].keys():
                                model_name_to_slide_names_to_embeddings[model_name][slide_name] = \
                                    flushed_model_name_to_slide_names_to_embeddings[model_name][slide_name]
                            else:
                                model_name_to_slide_names_to_embeddings[model_name][slide_name] += \
                                    flushed_model_name_to_slide_names_to_embeddings[model_name][slide_name]

            for gpu_id in range(1, self.world_size):
                os.remove(f"embeddings_rank_{gpu_id}.pkl")

            # DEBUG
            for model_name in model_name_to_slide_names_to_embeddings.keys():
                for slide_name in model_name_to_slide_names_to_embeddings[model_name].keys():
                    if len(model_name_to_slide_names_to_embeddings[model_name][slide_name]) != 500:
                        print(f"==> {model_name} {slide_name}"
                              f" {len(model_name_to_slide_names_to_embeddings[model_name][slide_name])}")

            for model_name in model_name_to_slide_names_to_embeddings.keys():
                checkpoint_id = model_name.split("/")[-1].split(".")[0]
                os.makedirs(f"embeddings/{self.run_name}/{checkpoint_id}", exist_ok=True)
                with open(f"embeddings/{self.run_name}/{checkpoint_id}/embeddings.pkl", "wb") as f:
                    pickle.dump(model_name_to_slide_names_to_embeddings[model_name], f)

    @staticmethod
    def _auc(
            model_name_to_models: Dict[str, DDP],
            data: DataLoader,
            patient_id_to_slide_name_and_label: dict,
            gpu_id: int,
            world_size: int,
            tested_epoch: int,
            dataset_name: str,
            run_name: str
    ):
        for _, model in model_name_to_models.items():
            model.eval()
        model_name_to_slide_name_to_probabilities = dict()
        device = f"cuda:{gpu_id}"

        for model_name, model in model_name_to_models.items():
            with (torch.no_grad()):
                for inputs, _, slide_name in tqdm(data, desc=f"Evaluate {dataset_name} AUC: GPU [{gpu_id}]"):
                    inputs = inputs.to(device)
                    output = model(inputs)
                    probability = softmax(output[0], dim=0)
                    cpu_probability = probability.cpu().numpy()
                    b_probability = cpu_probability.tolist()[1]

                    if model_name not in model_name_to_slide_name_to_probabilities.keys():
                        model_name_to_slide_name_to_probabilities[model_name] = dict()

                    if slide_name[0] not in model_name_to_slide_name_to_probabilities[model_name].keys():
                        model_name_to_slide_name_to_probabilities[model_name][slide_name[0]] = [b_probability]
                    else:
                        model_name_to_slide_name_to_probabilities[model_name][slide_name[0]].append(b_probability)

                if gpu_id != 0:
                    with open(f"{run_name}_{model_name}_outputs_rank_{gpu_id}.pkl", "wb") as f:
                        pickle.dump(model_name_to_slide_name_to_probabilities, f)

        barrier()

        if gpu_id == 0:
            for model_name, model in model_name_to_models.items():

                for gpu_id in range(1, world_size):
                    with open(f"{run_name}_{model_name}_outputs_rank_{gpu_id}.pkl", "rb") as f:
                        flushed_slide_name_to_probabilities = pickle.load(f)[model_name]
                        for slide_name in flushed_slide_name_to_probabilities.keys():
                            if model_name not in model_name_to_slide_name_to_probabilities.keys():
                                model_name_to_slide_name_to_probabilities[model_name] = dict()
                            if slide_name in model_name_to_slide_name_to_probabilities[model_name].keys():
                                model_name_to_slide_name_to_probabilities[model_name][slide_name].extend(
                                    flushed_slide_name_to_probabilities[slide_name])
                            else:
                                model_name_to_slide_name_to_probabilities[model_name][slide_name] = \
                                    flushed_slide_name_to_probabilities[
                                        slide_name]
                for gpu_id in range(1, world_size):
                    os.remove(f"{run_name}_{model_name}_outputs_rank_{gpu_id}.pkl")

                patient_id_to_mean_and_score_dict = dict()

                for patient_id in patient_id_to_slide_name_and_label.keys():
                    patient_slide_names_and_labels = patient_id_to_slide_name_and_label[patient_id]
                    means = []
                    for slide_name, label in patient_slide_names_and_labels:
                        probabilities_mean = np.array(
                            model_name_to_slide_name_to_probabilities[model_name][slide_name]).mean()
                        means.append(probabilities_mean)
                    patient_mean = np.array(means).mean()
                    patient_id_to_mean_and_score_dict[patient_id] = {
                        "mean": patient_mean,
                        "label": label
                    }
                auc_score, l_interval, r_interval = confidence_interval(patient_id_to_mean_and_score_dict)

                # if os.path.isfile(f"stats/{run_name}/{model_name}_{dataset_name}_auc_scores.pkl"):
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_auc_scores.pkl", "rb") as f:
                #         auc_scores = pickle.load(f)
                #     auc_scores.append(auc_score)
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_auc_scores.pkl", "wb") as f:
                #         pickle.dump(auc_scores, f)
                # else:
                #     auc_scores = [auc_score]
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_auc_scores.pkl", "wb") as f:
                #         pickle.dump(auc_scores, f)
                #
                # if os.path.isfile(f"stats/{run_name}/{model_name}_{dataset_name}_epochs.pkl"):
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_epochs.pkl", "rb") as f:
                #         epochs = pickle.load(f)
                #     epochs.append(tested_epoch)
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_epochs.pkl", "wb") as f:
                #         pickle.dump(epochs, f)
                # else:
                #     epochs = [tested_epoch]
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_epochs.pkl", "wb") as f:
                #         pickle.dump(epochs, f)
                #
                # if os.path.isfile(f"stats/{run_name}/{model_name}_{dataset_name}_confidence_interval.pkl"):
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_confidence_interval.pkl", "rb") as f:
                #         confidence_intervals = pickle.load(f)
                #     confidence_intervals.append((l_interval, r_interval))
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_confidence_interval.pkl", "wb") as f:
                #         pickle.dump(confidence_intervals, f)
                # else:
                #     confidence_intervals = [(l_interval, r_interval)]
                #     with open(f"stats/{run_name}/{model_name}_{dataset_name}_confidence_interval.pkl", "wb") as f:
                #         pickle.dump(confidence_intervals, f)

                print(
                    f"==> for {model_name} AUC {dataset_name} score for Epoch[{tested_epoch}]:"
                    f" {auc_score}; ({l_interval}, {r_interval})"
                )

                if dataset_name == "test":
                    wandb.log({"test_auc_score": auc_score, "epoch": tested_epoch})
                else:
                    wandb.log({"train_auc_score": auc_score, "epoch": tested_epoch})

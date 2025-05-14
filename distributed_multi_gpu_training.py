import os
import sys
import torch
import torch.multiprocessing as mp
import wandb
import shutil
from typing import Dict, List
from train.trainer import Trainer
from eval.evaluator import Evaluator
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from datasets.utils import prepare_dataloader, load_train_objs, load_evaluate_objs
from utils.utils import validate_dataset_with_csv, parse_json_input, ddp_setup, wandb_setup, get_free_port
from torch.distributed import destroy_process_group, barrier
from datasource.datasource import Datasource


def train(
        rank: int,
        world_size: int,
        img_dirs: list,
        grid_files_path: list,
        train_annotations_path: str,
        test_annotations_path: str,
        train_epochs_cnt: int,
        train_batch_size: int,
        train_mode: str,
        log_train_auc: bool,
        log_test_auc: bool,
        log_slide_scores: bool,
        lr: float,
        weight_decay: float,
        model_save_frequency: int,
        checkpoints_to_upload: list,
        project_name: str,
        run_name: str,
        train_auc_frequency: int,
        test_auc_frequency: int,
        preload_model: bool,
        preloaded_model_path: str,
        objective: str,
        datasource_name: str,
        architecture: str,
        free_port_ddp: str
):
    ddp_setup(rank, world_size, free_port_ddp)

    if rank == 0:
        wandb_setup(project_name, run_name, objective, architecture, train_epochs_cnt, train_batch_size, lr, weight_decay)
    barrier()

    train_dataset: Dataset
    train_auc_dataset: Dataset
    test_auc_dataset: Dataset
    slide_scores_dataset: Dataset

    train_dataloader: DataLoader
    train_auc_dataloader = None
    slide_scores_dataloader = None

    model: models.ResNet
    optimizer: torch.optim.Optimizer

    trainer: Trainer
    evaluator: Evaluator

    train_dataset, model, optimizer = load_train_objs(
        rank=rank,
        img_dirs=img_dirs,
        architecture=architecture,
        grid_files_path=grid_files_path,
        train_annotations_path=train_annotations_path,
        train_mode=train_mode,
        lr=lr,
        weight_decay=weight_decay,
        preload_model=preload_model,
        preloaded_model_path=preloaded_model_path
    )

    train_dataloader = prepare_dataloader(train_dataset, train_batch_size)

    if log_train_auc:
        train_auc_dataset = load_evaluate_objs(
            img_dirs=img_dirs,
            grid_files_path=grid_files_path,
            annotations_path=train_annotations_path,
            evaluate_mode="random",
            sample_size=100
        )
        train_auc_dataloader = prepare_dataloader(train_auc_dataset, 1)

    test_auc_dataset = load_evaluate_objs(
        img_dirs=img_dirs,
        grid_files_path=grid_files_path,
        annotations_path=test_annotations_path,
        evaluate_mode="random",
        sample_size=100
    )
    test_auc_dataloader = prepare_dataloader(test_auc_dataset, 1)

    if log_slide_scores:
        slide_scores_dataset = load_evaluate_objs(
            img_dirs=img_dirs,
            grid_files_path=grid_files_path,
            annotations_path=test_annotations_path,
            evaluate_mode="random",
            sample_size=500
        )
        slide_scores_dataloader = prepare_dataloader(slide_scores_dataset, 1)

    trainer = Trainer(model,
                      train_dataloader,
                      optimizer,
                      rank,
                      run_name,
                      objective,
                      datasource_name,
                      checkpoints_to_upload
                      )
    evaluator = Evaluator({"main": trainer.model},
                          rank,
                          world_size,
                          train_annotations_path,
                          test_annotations_path,
                          train_auc_dataloader,
                          test_auc_dataloader,
                          slide_scores_dataloader,
                          None,
                          log_train_auc,
                          run_name
                          )

    for epoch in range(train_epochs_cnt):
        trainer.run_epoch(epoch)

        if rank == 0 and (epoch + 1) % model_save_frequency == 0:
            trainer.save_checkpoint(epoch + 1)

        if log_train_auc and (epoch + 1) % train_auc_frequency == 0:
            barrier()
            evaluator.evaluate_train_auc(epoch + 1)
            barrier()
            trainer.model.train()

        if log_test_auc and (epoch + 1) % test_auc_frequency == 0:
            barrier()
            evaluator.evaluate_test_auc(epoch + 1)
            barrier()
            trainer.model.train()

    barrier()
    if log_train_auc:
        evaluator.evaluate_train_auc(train_epochs_cnt)
    barrier()
    evaluator.evaluate_test_auc(train_epochs_cnt)
    barrier()
    if log_slide_scores:
        evaluator.evaluate_slide_scores()
    if rank == 0:
        wandb.finish()
    barrier()
    destroy_process_group()


def main(input_dict: dict):
    datasource_train: str = input_dict["datasource_train"]
    datasource_test: str = input_dict["datasource_test"]
    folds: List[str] = input_dict["folds"]
    objective: str = input_dict["objective"]
    architecture: str = input_dict["architecture"]

    train_epochs_cnt: int = int(input_dict["train_epochs_count"])
    train_batch_size: int = int(input_dict["train_batch_size"])
    train_mode: str = input_dict["train_mode"]
    log_train_auc: bool = eval(input_dict["log_train_auc"])
    log_test_auc: bool = eval(input_dict["log_test_auc"])
    log_slide_scores: bool = eval(input_dict["log_slide_scores"])

    lr: float = float(input_dict["lr"])
    weight_decay: float = float(input_dict["weight_decay"])

    model_save_frequency: int = int(input_dict["model_save_frequency"])
    checkpoints_to_upload: List[str] = input_dict["checkpoints_to_upload"]
    project_name: str = input_dict["project_name"]
    run_name: str = input_dict["run_name"]
    train_auc_frequency: int = int(input_dict["train_auc_frequency"])
    test_auc_frequency: int = int(input_dict["test_auc_frequency"])

    fine_tune_mode: bool = eval(input_dict["fine_tune_mode"])
    fine_tune_epochs_count: int = int(input_dict["fine_tune_epochs_count"])
    models_dirs: Dict[str, str] = input_dict["models_dirs"]
    checkpoints_to_fine_tune: List[str] = input_dict["checkpoints_to_fine_tune"]

    world_size = torch.cuda.device_count()

    print(f"==> Multi GPU training/testing with {world_size} GPUs will commence with the following parameters:")
    print(f"==> world_size:", world_size)
    print(f"==> Total train epochs:", train_epochs_cnt)
    print(f"==> Train batch size:", train_batch_size)
    print(f"==> Train mode:", train_mode)
    print(f"==> Learning rate:", lr)
    print(f"==> Train AUC frequency:", train_auc_frequency)
    print(f"==> Test AUC frequency:", test_auc_frequency)
    print(f"==> Fine Tune mode:", fine_tune_mode)
    print(f"==> Fine Tune epochs:", fine_tune_epochs_count)

    for fold in folds:
        if fine_tune_mode:
            models_dir = models_dirs[fold]
        else:
            models_dir = ""

        ds = Datasource(
            datasource_train=datasource_train,
            datasource_test=datasource_test,
            train_fold=fold,
            test_fold=fold,
            objective=objective,
            models_dir=models_dir,
        )

        print(f"validate datsource test {ds.test_annotations_file_path}")
        missing_names = validate_dataset_with_csv(ds.img_dirs,
                                                  ds.train_annotations_file_path,
                                                  ds.test_annotations_file_path)

        print("passed validation")
        if missing_names:
            print(f"==> Missing:\n {missing_names}")
            sys.exit(1)

        if fine_tune_mode:
            checkpoints = ds.get_list_of_models_path(checkpoints_to_fine_tune)
            epochs = fine_tune_epochs_count
        else:
            checkpoints = [""]
            epochs = train_epochs_cnt
        for checkpoint in checkpoints:
            if checkpoint == "":
                run_name_suffix = f"_{fold}"
            else:
                run_name_suffix = f"_{fold}_{checkpoint[-7:-3]}"

            free_port_ddp = get_free_port()

            mp.spawn(train,
                     args=(
                         world_size,
                         ds.img_dirs,
                         ds.grid_files_path,
                         ds.train_annotations_file_path,
                         ds.test_annotations_file_path,
                         epochs,
                         train_batch_size,
                         train_mode,
                         log_train_auc,
                         log_test_auc,
                         log_slide_scores,
                         lr,
                         weight_decay,
                         model_save_frequency,
                         checkpoints_to_upload,
                         project_name,
                         run_name + run_name_suffix,
                         train_auc_frequency,
                         test_auc_frequency,
                         fine_tune_mode,
                         f"checkpoints/{checkpoint}",
                         objective,
                         datasource_train,
                         architecture,
                         free_port_ddp
                     ),
                     nprocs=world_size)

        if fine_tune_mode:
            os.makedirs(f"checkpoints/{datasource_train.lower()}/{objective}/{run_name}_{fold}", exist_ok=True)
            for checkpoint in checkpoints:
                shutil.copy(
                    f"checkpoints/{datasource_train.lower()}/{objective}/{run_name}_{fold}_{checkpoint[-7:-3]}"
                    f"/checkpoint_{fine_tune_epochs_count}.pt",
                    f"checkpoints/{datasource_train.lower()}/{objective}/{run_name}_{fold}"
                    f"/checkpoint_{checkpoint[-7:-3]}.pt")


if __name__ == "__main__":
    main(parse_json_input(sys.argv))

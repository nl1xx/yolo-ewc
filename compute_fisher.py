import torch
import argparse
from tqdm import tqdm
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.models.yolo.pose.train import PoseTrainer

TRAINER_REGISTRY = {
    'detect': DetectionTrainer,
    'segment': SegmentationTrainer,
    'classify': ClassificationTrainer,
    'pose': PoseTrainer,
}


def compute_fisher_information(model_path, data_config_path, save_path, device='cuda'):
    print("Starting Fisher Information Computation (Simplified Version)")
    print(f"Model path: {model_path}")
    print(f"Data config: {data_config_path}")
    print(f"Device: {device}")

    try:
        # 1. Determine the task from the model file
        print("Loading model to determine task type.")
        model_obj = YOLO(model_path)
        task = model_obj.task
        print(f"Task automatically identified as: '{task}'")

        if task not in TRAINER_REGISTRY:
            raise NotImplementedError(f"Task '{task}' not in TRAINER_REGISTRY.")
        TrainerClass = TRAINER_REGISTRY[task]
        print(f"Found corresponding trainer: {TrainerClass.__name__}")

        # 2. Prepare configuration overrides
        overrides = {'model': model_path, 'data': data_config_path, 'mode': 'train', 'device': device}

        # 3. Instantiate the trainer and run its setup to get a valid dataloader and model
        print("Setting up trainer to get dataloader and a configured model.")
        trainer = TrainerClass(overrides=overrides)
        trainer._setup_train(world_size=1)

        # 4. Extract the necessary components
        dataloader = trainer.train_loader
        model = trainer.model
        model.train()

    except Exception as e:
        print(f"\nERROR: Failed to set up the trainer.")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Trainer and dataloader setup successful. Number of batches: {len(dataloader)}")

    # 5. Initialize the Fisher dictionary
    fisher_dict = {name: torch.zeros_like(param.data) for name, param in model.named_parameters() if param.requires_grad}

    # 6. Compute the Fisher Information Matrix
    print("Computing Fisher Information Matrix.")
    num_samples = 0
    for batch in tqdm(dataloader, desc="Calculating Fisher Information"):
        batch = trainer.preprocess_batch(batch)
        model.zero_grad()

        loss, _ = model(batch)

        loss.sum().backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data.clone().pow(2)

        num_samples += len(batch['img'])

    # 7. Average the Fisher values
    print("Finalizing Fisher matrix.")
    if num_samples == 0:
        print("ERROR: No samples were processed from the dataloader.")
        return

    for name in fisher_dict:
        fisher_dict[name] /= num_samples

    # 8. Store the optimal parameters
    optimal_params = {name: param.data.clone() for name, param in model.named_parameters()}

    # 9. Save the EWC data
    print(f"Step 5: Saving EWC data to '{save_path}'...")
    torch.save({'fisher': fisher_dict, 'optimal_params': optimal_params}, save_path)

    print("Fisher Information Computation Finished Successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Fisher Information Matrix for any YOLO task.")
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained .pt model file (e.g., yolo11n.pt, yolo11n-seg.pt).")
    parser.add_argument('--data', type=str, required=True, help="Path to the dataset's .yaml configuration file.")
    parser.add_argument('--save-path', type=str, required=True, help="Path to save the output EWC data file.")
    parser.add_argument('--device', type=str, default='cuda', help="Device for computation ('cuda' or 'cpu').")

    args = parser.parse_args()

    compute_fisher_information(
        model_path=args.model,
        data_config_path=args.data,
        save_path=args.save_path,
        device=args.device
    )

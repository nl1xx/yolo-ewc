import sys
from ultralytics.models.yolo import YOLO
from ultralytics.engine.ewc import EWC
from compute_fisher import TRAINER_REGISTRY


def main():
    """
    Parses all arguments, separates custom from standard ones, creates a standard
    trainer, and then injects the custom EWC logic before starting training.
    """
    args_list = sys.argv[1:]

    # Separate standard YOLO args from our custom EWC args
    yolo_args = {}
    ewc_params = {}
    for arg in args_list:
        if '=' in arg:
            key, value = arg.split('=', 1)
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            # Separate the arguments
            if key in ['ewc_data', 'ewc_lambda']:
                ewc_params[key] = value
            else:
                yolo_args[key] = value

    print("--- Starting Training with Isolated Custom Arguments ---")
    print(f"Standard YOLO Arguments: {yolo_args}")
    print(f"Custom EWC Arguments: {ewc_params}")

    # reate the correct Trainer using ONLY standard args
    model_path = yolo_args.get('model', 'yolo11n.pt')
    task = YOLO(model_path).task
    if task not in TRAINER_REGISTRY:
        raise ValueError(f"Task '{task}' not supported by the TRAINER_REGISTRY in compute_fisher.py")

    TrainerClass = TRAINER_REGISTRY[task]

    # Instantiate the trainer with only the official YOLO arguments
    trainer = TrainerClass(overrides=yolo_args)

    # Manually inject the EWC logic into the trainer instance
    ewc_data_str = ewc_params.get('ewc_data')
    ewc_lambda_val = ewc_params.get('ewc_lambda', 0.0)

    trainer.ewc = None
    if ewc_data_str and ewc_lambda_val > 0.0:
        print("\nInjecting EWC logic into the trainer...")
        ewc_data_paths = [path.strip() for path in ewc_data_str.split(',')]

        trainer.setup_model()
        trainer.model = trainer.model.to(trainer.device)

        trainer.ewc = EWC(trainer.model, ewc_data_paths, ewc_lambda_val, trainer.device)
        print("EWC logic successfully injected.")

    # Start the training
    # We call trainer.train() directly, not model.train()
    trainer.train()

    print("\n--- Training Finished ---")


if __name__ == '__main__':
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    from ultralytics.models.yolo.classify.train import ClassificationTrainer
    from ultralytics.models.yolo.pose.train import PoseTrainer

    main()

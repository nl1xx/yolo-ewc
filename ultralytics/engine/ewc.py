import torch


class EWC:
    """
    Elastic Weight Consolidation (EWC) helper class for multitask continual learning.

    This class loads EWC data (Fisher Information Matrix and optimal parameters) from
    one or more previous tasks and computes a cumulative penalty to prevent
    catastrophic forgetting.
    """

    def __init__(self, model, ewc_data_paths: list, ewc_lambda=1.0, device='cuda'):
        """
        Initializes the multitask EWC module.
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device

        # Store lists of Fisher matrices and optimal parameters for each past task
        self.fishers = []
        self.optimal_params_list = []
        self.enabled = False

        if not isinstance(ewc_data_paths, list):
            print(f"WARNING: ewc_data_paths is not a list. EWC will be disabled.")
            return

        print(f"INFO: EWC initializing for {len(ewc_data_paths)} previous task(s).")
        for i, path in enumerate(ewc_data_paths):
            try:
                print(f" Loading EWC data for task {i + 1} from '{path}'")
                data = torch.load(path, map_location=device)

                fisher = {name: p.to(self.device) for name, p in data['fisher'].items()}
                optimal_params = {name: p.to(self.device) for name, p in data['optimal_params'].items()}

                self.fishers.append(fisher)
                self.optimal_params_list.append(optimal_params)
                print(f"Successfully loaded data for task {i + 1}.")
            except FileNotFoundError:
                print(f"WARNING: EWC data file not found at '{path}'. Skipping this task.")
            except Exception as e:
                print(f"ERROR: Failed to load EWC data from '{path}'. Skipping. Error: {e}")

        if self.fishers and self.optimal_params_list:
            self.enabled = True
            print(f"INFO: EWC enabled for {len(self.fishers)} task(s).")
        else:
            print("WARNING: No valid EWC data was loaded. EWC is disabled.")

    def compute_penalty(self):
        """
        Calculates the cumulative EWC penalty across all loaded past tasks.

        The total penalty is the sum of penalties for each individual past task.
        L_ewc = (lambda / 2) * sum_over_tasks( sum_over_params( F_i * (theta_i - theta_i*)^2 ) )
        """
        if not self.enabled:
            return torch.tensor(0.0, device=self.device)

        total_penalty = 0.0

        # Iterate through each past task's data
        for task_idx, (fisher, optimal_params) in enumerate(zip(self.fishers, self.optimal_params_list)):
            task_penalty = 0.0
            # Iterate through the parameters of the current model
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in fisher:
                    # Calculate the squared difference, weighted by the Fisher value for this task
                    diff = (param - optimal_params[name]).pow(2)
                    weighted_diff = fisher[name] * diff

                    # Sum up the penalty for the current parameter
                    task_penalty += weighted_diff.sum()

            # Add the penalty for this task to the total penalty
            total_penalty += task_penalty

        return (self.ewc_lambda / 2.0) * total_penalty

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss


class nnUNetTrainerPituitary(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1500
        self.initial_lr = 5e-4
        self.oversample_foreground_percent = 1.0
        print("nnUNetTrainerPituitary: 1500 epochs, weighted loss, 100% FG oversample")

    def _build_loss(self):
        class_weights = torch.tensor(
            [0.1, 1.0, 3.0, 3.5, 3.5, 4.0], dtype=torch.float32
        )
        return DC_and_CE_loss(
            {"batch_dice": self.configuration_manager.batch_dice,
             "smooth": 1e-5, "do_bg": False, "ddp": self.is_ddp},
            {"weight": class_weights},
            weight_ce=1.0, weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), self.initial_lr,
            weight_decay=self.weight_decay, momentum=0.95, nesterov=True,
        )
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(
            optimizer,
            lambda epoch: (1 - epoch / self.num_epochs) ** 0.9
        )
        return optimizer, scheduler

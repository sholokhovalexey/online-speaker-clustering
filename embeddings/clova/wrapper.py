import torch
from .voxceleb_trainer.models.ResNetSE34V2 import MainModel

from pretrained import model_path_clova


def prepare_model_clova(model_path=model_path_clova, device="cuda:0", n_mels=64):

    clova_state = torch.load(model_path)
    model = MainModel(nOut=512, encoder_type="ASP", n_mels=n_mels)
    default_state = model.state_dict()

    for name, param in clova_state.items():
        if "__S__" in name:
            valid_name = name[6:]
            if valid_name not in default_state:
                continue

            if default_state[valid_name].size() != param.size():
                print(
                    f"mismatched size: model: {default_state[valid_name].size()}; state: {param.size()}"
                )
                continue

            default_state[valid_name].copy_(param)

    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return model

import os
from huggingface_hub import snapshot_download


MODEL_DICT = {
    "Woodblock": "BDRC/Woodblock",
    "DergeTenjur": "BDRC/DergeTenjur",
    "GoogleBooks_C": "BDRC/GoogleBooks_C_v1",
    "GoogleBooks_E": "BDRC/GoogleBooks_E_v1",
}


# download the line model: https://huggingface.co/BDRC/PhotiLines
def init_monlam_line_model() -> str:
    model_id = "BDRC/PhotiLines"
    model_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=f"Models/{model_id}",
    )
    model_config = f"{model_path}/config.json"
    assert os.path.isfile(model_config)

    return model_config


# download the layout model: https://huggingface.co/BDRC/Photi
def init_monlam_layout_model() -> str:
    model_id = "BDRC/Photi" 
    model_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=f"Models/{model_id}",
    )

    model_config = f"{model_path}/config.json"
    assert os.path.isfile(model_config)

    return model_config


def init_monla_ocr_model(identifier: str) -> str:
    available_models = list(MODEL_DICT.keys())

    if identifier in available_models:
        model_id = MODEL_DICT[identifier]

        model_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_dir=f"Models/{model_id}",
        )

        model_config = f"{model_path}/config.json"
        assert os.path.isfile(model_config)

        return model_config
    else:
        print(f"Error: {identifier} is not available")
        return None

import wandb

from src.config.config import Config


def apply_wandb_overrides(config: Config) -> Config:
    for key in wandb.config.keys():
        if "." not in key:
            continue

        section, field_name = key.split(".", 1)

        if hasattr(config, section):
            section_config = getattr(config, section)
            if hasattr(section_config, field_name):
                setattr(section_config, field_name, wandb.config[key])

    config.model.__post_init__()

    return config

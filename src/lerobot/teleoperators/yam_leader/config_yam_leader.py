from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("yam_leader")
@dataclass
class YAMLeaderConfig(TeleoperatorConfig):
    # Port to connect to the arm
    port: str

    use_degrees: bool = False
from typing import Literal

VALID_CALIBRATION_TYPES = Literal[
    "scp_task_thresholds",
    "scp_global_threshold",
    "ccp_class_thresholds",
    "ccp_task_cluster_thresholds",
    "ccp_global_cluster_thresholds",
    "ccp_joint_class_repr",
]

VALID_NONCONFORMITY_METHODS = Literal[
    "hinge",
    "margin",
    "pip",
]

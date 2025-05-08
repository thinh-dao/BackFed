"""
Utility functions for FL.
"""

from fl_bdbench.utils.model_utils import (
    get_model,
    get_layer_names,
    get_normalization,
    get_last_layer_name
)

from fl_bdbench.utils.system_utils import (
    system_startup,
    set_attack_config,
    set_random_seed,
    set_debug_settings,
    pool_size_from_resources
)

from fl_bdbench.utils.logging_utils import (
    log,
    CSVLogger,
    get_console,
    init_csv_logger,
    init_wandb,
    plot_csv,
    save_model_to_wandb_artifact
)

from fl_bdbench.utils.server_utils import (
    test,
    clip_updates_inplace,
    clip_updates,
    model_dist_layer
)

from fl_bdbench.utils.misc_utils import (
    sync_to_async,
    with_timeout,
    format_time_hms
)

# For backward compatibility, re-export everything at the top level
__all__ = [
    # Model utilities
    'get_model', 'get_layer_names', 'get_normalization', 'get_last_layer_name',
    # System utilities
    'system_startup', 'set_attack_config', 'set_random_seed', 'set_debug_settings', 'pool_size_from_resources',
    # Logging utilities
    'log', 'CSVLogger', 'get_console', 'init_wandb', 'init_csv_logger', 'plot_csv', 'save_model_to_wandb_artifact',
    # Server utilities
    'test', 'clip_updates_inplace', 'clip_updates', 'model_dist_layer',
    # Misc utilities
    'sync_to_async', 'with_timeout', 'format_time_hms'
]

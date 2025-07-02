"""
Utility functions for FL.
"""

from backfed.utils.model_utils import (
    get_model,
    get_layer_names,
    get_normalization,
    get_last_layer_name
)

from backfed.utils.system_utils import (
    system_startup,
    set_attack_config,
    set_random_seed,
    set_debug_settings,
    pool_size_from_resources
)

from backfed.utils.logging_utils import (
    log,
    CSVLogger,
    get_console,
    init_csv_logger,
    init_wandb,
    plot_csv,
    save_model_to_wandb_artifact
)

from backfed.utils.server_utils import (
    clip_updates_inplace,
    clip_updates,
    model_dist_layer,
    test_classifier,
    test_lstm_reddit,
)

from backfed.utils.misc_utils import (
    sync_to_async,
    with_timeout,
    format_time_hms
)

from backfed.utils.text_utils import (
    Dictionary,
    get_tokenizer,
    get_batches,
    get_word_list,
    batchify,
    repackage_hidden,
    get_word_list
)

__all__ = [
    # Model utilities
    'get_model', 'get_layer_names', 'get_normalization', 'get_last_layer_name',
    # System utilities
    'system_startup', 'set_attack_config', 'set_random_seed', 'set_debug_settings', 'pool_size_from_resources',
    # Logging utilities
    'log', 'CSVLogger', 'get_console', 'init_wandb', 'init_csv_logger', 'plot_csv', 'save_model_to_wandb_artifact',
    # Server utilities
    'clip_updates_inplace', 'clip_updates', 'model_dist_layer', 'test_classifier', 'test_lstm_reddit',
    # Misc utilities
    'sync_to_async', 'with_timeout', 'format_time_hms',
    # Text utilities
    'Dictionary', 'get_tokenizer', 'get_batches', 'get_word_list', 'batchify', 'repackage_hidden'
]

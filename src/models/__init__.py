import jax
from jaxtyping import PyTree
import equinox as eqx

from models.lstm_mlp_attn import LSTM_MLP_ATTN
from models.stacked_lstm import STACKED_LSTM
from models.ea_lstm import EA_LSTM
from models.ea_transformer import EA_TRANSFORMER


def make(cfg: dict):
    """Creates a model based on the provided configuration.

    Parameters
    ----------
    cfg: dict
        A dictionary containing the configuration for the model.

    Returns
    -------
    eqx.Module
        The created model.
    """
    name = cfg["model"].lower()
    if name == "lstm_mlp_attn":
        model_fn = LSTM_MLP_ATTN
    elif name == "ea_lstm":
        model_fn = EA_LSTM
    elif name == "stacked_lstm":
        model_fn = STACKED_LSTM
    elif name == "ea_transformer":
        model_fn = EA_TRANSFORMER
    else:
        err_str = (
            f"{cfg['model']} is not a valid model name. "
            + "Check /src/models/__init__.py for model config."
        )
        raise ValueError(err_str)

    model = model_fn(**cfg["model_args"])
    num_params, memory_bytes = _count_parameters(model)
    size, unit = _human_readable_size(memory_bytes)
    print(f"Model contains {num_params:,} parameters, using {size:.2f}{unit} memory.")
    return model


def _count_parameters(model: PyTree):
    """Counts the trainable parameters in a model and estimates its memory usage.

    Parameters
    ----------
    model: PyTree
        Model to count parameters.

    Returns
    -------
    num_params: int
        Total number of trainable parameters in the model.
    memory_bytes: int
        Estimated memory usage of the model in bytes, assuming each parameter is a
        32-bit float.
    """
    # Use tree_flatten to get a list of arrays and ensure is_leaf treats arrays as leaves
    params, _ = jax.tree_util.tree_flatten(model)
    # Count the total number of parameters
    num_params = sum(param.size for param in params if eqx.is_inexact_array(param))
    # Calculate memory usage assuming 4 bytes per parameter (32-bit float)
    memory_bytes = num_params * 4

    return num_params, memory_bytes


# Convert bytes to a human-readable format
def _human_readable_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            break
        size /= 1024.0
    return size, unit

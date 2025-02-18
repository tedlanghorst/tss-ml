import jax
import equinox as eqx

# from models.lstm import TAPLSTM
# from models.transformer import EATransformer
# from models.hybrid import Hybrid
from models.flexible_hybrid import FlexibleHybrid
# from models.tft import TFT
# from models.tft_mha import TFT_MHA
from models.lstm_mlp_attn import LSTM_MLP_ATTN
from models.rg_lstm import Graph_LSTM

def make(cfg: dict):
    name = cfg['model'].lower()
    if name == "flexible_hybrid":
        model_fn = FlexibleHybrid
    elif name == "lstm_mlp_attn":
        model_fn = LSTM_MLP_ATTN
    elif name == 'graph_lstm':
        model_fn = Graph_LSTM
    else:
        raise ValueError(f"{cfg['model']} is not a valid model name. Check /src/models/__init__.py for model config.")

    model = model_fn(**cfg['model_args'])
    num_params, memory_bytes = count_parameters(model)
    size, unit = human_readable_size(memory_bytes)
    print(f"Model contains {num_params:,} parameters, using {size:.2f}{unit} memory.")
    return model

def count_parameters(model):
    # Use tree_flatten to get a list of arrays and ensure is_leaf treats arrays as leaves
    params, _ = jax.tree_util.tree_flatten(model)
    # Count the total number of parameters
    num_params = sum(param.size for param in params if eqx.is_inexact_array(param))
    # Calculate memory usage assuming 4 bytes per parameter (32-bit float)
    memory_bytes = num_params * 4

    return num_params, memory_bytes

# Convert bytes to a human-readable format
def human_readable_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0 or unit == 'TB':
            break
        size /= 1024.0
    return size, unit
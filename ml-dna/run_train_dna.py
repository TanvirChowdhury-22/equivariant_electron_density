from e3nn.nn.models import gate_points_2101
from e3nn import o3

# Save original __init__ function
_original_init = gate_points_2101.Network.__init__

def patched_init(self, **kwargs):
    # Convert list-based irreps_hidden to string-based if needed
    if isinstance(kwargs.get("irreps_hidden"), list):
        mapping = {-1: 'o', 1: 'e'}  # Correct parity mapping
        parts = []
        for mul, l, p in kwargs["irreps_hidden"]:
            parts.append(f"{mul}x{l}{mapping[p]}")
        kwargs["irreps_hidden"] = " + ".join(parts)
    _original_init(self, **kwargs)

gate_points_2101.Network.__init__ = patched_init

# Run the training script
import train_dna


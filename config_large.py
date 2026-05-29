"""Large dataset configuration — more data for neural models.

- n_users_sample: 5000 → 20000
- n_items_target: 3000 → 10000
"""

from config import CONFIG as _BASE

CONFIG = _BASE.copy()
CONFIG["filter"] = {**_BASE["filter"],
    "n_users_sample": 20000,
    "n_items_target": 10000,
}

# Distrax Tabulate

> Allows to tabulate Flax modules that use distrax Distributions

- [Distrax Tabulate](#distrax-tabulate)
  - [Example](#example)
  - [How it works](#how-it-works)


## Example

```python
import distrax as dx
import flax.linen as nn
import jax
import jax.numpy as jnp

#### Import the module and run the function ####
from dx_tabulate import add_distrax_representers

add_distrax_representers()
################################################


class Policy(nn.Module):
    @nn.compact
    def __call__(self, x):
        logits = nn.Dense(10)(x)
        return dx.Categorical(logits)


tabulate_fn = nn.tabulate(
    Policy(), jax.random.key(0), compute_flops=True, compute_vjp_flops=True
)
print(tabulate_fn(jnp.ones((1, 15))))
```

```bash
                                         Policy Summary                                          
┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ path    ┃ module ┃ inputs        ┃ outputs       ┃ flops ┃ vjp_flops ┃ params                 ┃
┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│         │ Policy │ float32[1,15] │ Categorical   │ 348   │ 1148      │                        │
├─────────┼────────┼───────────────┼───────────────┼───────┼───────────┼────────────────────────┤
│ Dense_0 │ Dense  │ float32[1,15] │ float32[1,10] │ 310   │ 1070      │ bias: float32[10]      │
│         │        │               │               │       │           │ kernel: float32[15,10] │
│         │        │               │               │       │           │                        │
│         │        │               │               │       │           │ 160 (640 B)            │
├─────────┼────────┼───────────────┼───────────────┼───────┼───────────┼────────────────────────┤
│         │        │               │               │       │     Total │ 160 (640 B)            │
└─────────┴────────┴───────────────┴───────────────┴───────┴───────────┴────────────────────────┘
                                                                                                 
                                  Total Parameters: 160 (640 B)                          
```

## How it works 

> [!TIP]
> Flax `tabulate` uses yaml to render its table.

The `add_distrax_representers` function first finds all subclasses of `distrax.Distribution` in the inheritance graph. Then it proceeds to add a yaml representer for all of them, using the `name` property.

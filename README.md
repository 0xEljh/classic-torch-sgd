# classic-torch-sgd
Pytorch SGD implementation that reverts to the original update formulas (Sutskever et. al.): more intuitive momentum and learning rate behaviour

As per Pytorch documentation (extracted from [1.13 docs](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)):
>The implementation of SGD with Momentum/Nesterov subtly differs from Sutskever et. al. and implementations in some other frameworks.<br><br>Considering the specific case of Momentum, the update can be written as<br> $$v_{t+1} = \mu*{v_t} + g_{t+1}$$ $$p_{t+1} = p_t - lr*v_{t+1}$$ where $p$, $g$, $v$, and $\mu$ denote the parameters, gradient, velocity, and momentum respectively.<br><br> This is in contrast to Sutskever et. al. and other frameworks which employ an update of the form<br> $$v_{t+} = \mu * v_t + lr * g_{t+1}$$ $$p_{t+1} = p_t - v_{t+1}$$ The Nesterov version is analogously modified.

The implementation of `ClassicSGD` hence follows the update method of the latter.

By modifying this update method, the implications of adjusting the learning rate and momentum terms become more intuitive and are easily separable:<br>
- Originally, the effect of the velocity, $v_t$ is modulated by both the momentum, $\mu$, and learning rate, $lr$. Now, only $\mu$ controls the size of the update due to velocity.
- Similarly, the learning rate term affects only the incoming gradient signal, $g_{t+1}$, which means the impact of the incoming gradient signal on velocity can be directly modulated (a task that previously had to be done through a convoluted weighing against $\mu$)

With these adjustments, it should be easier to tune and understand the implcations of the learning rate and momentum settings in the SGD optimizer.

## Why This Matters for Learning Rate Scheduling

The decoupling becomes particularly important when using learning rate schedulers like `ReduceLROnPlateau`:

| Event | PyTorch SGD | ClassicSGD |
|-------|-------------|------------|
| LR reduced | Accumulated velocity's impact suddenly scaled down (p = p - **lr**\*v) | Velocity continues at current scale; only new gradients contribute less |
| Effect | Can cause abrupt changes in training dynamics | Smoother transitions, more predictable behavior |

## Comparison Script

A head-to-head comparison script is included to demonstrate these differences empirically.

### Installation

```bash
pip install -r requirements.txt
```

### Running the Comparison

```bash
# Default settings (40 epochs, FashionMNIST)
python compare_optimizers.py

# Custom configuration
python compare_optimizers.py --epochs 60 --lr 0.1 --momentum 0.95 --patience 5

# Headless mode (no plot display, saves to file)
python compare_optimizers.py --no-plot --save-plot results.png
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 40 | Number of training epochs |
| `--lr` | 0.05 | Initial learning rate |
| `--momentum` | 0.9 | Momentum coefficient |
| `--batch-size` | 128 | Training batch size |
| `--patience` | 3 | ReduceLROnPlateau patience |
| `--factor` | 0.5 | LR reduction factor |
| `--seed` | 42 | Random seed for reproducibility |
| `--no-plot` | False | Skip plot display |
| `--save-plot` | comparison_results.png | Output path for plot |

### What the Comparison Shows

The script trains identical CNN models on FashionMNIST using both optimizers with the same:
- Initial weights (cloned model)
- Hyperparameters (LR, momentum)
- Data ordering (seeded shuffle)
- LR scheduler settings

This isolates the optimizer formulation as the only variable, allowing direct observation of how each handles learning rate reductions during training.

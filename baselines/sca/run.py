import torch
import numpy as np

from baselines.sca.model import USCA


def run_usca(X_train, Y_train, output_size, X_test=None, Y_test=None, **kwargs):
    """
    Run a single USCA training + transform cycle.

    Args:
        X_train, Y_train: Training tensors.
        output_size (int or list[int]): Latent dimension(s). If a list,
            the full training + inference is repeated for each dim and
            results are returned as dict {dim: (out1, out2)}.
        X_test, Y_test: Test tensors (default to train if None).
        **kwargs: Any additional USCA / training parameters. Supported keys:
            Model:  orthogonal_w, supervised_w, encoder_lr, discr_lr,
                    batch_size, n_z, lsmooth, device, verbose, loss_func
            Training: num_anchors, n_epochs

    Returns:
        (output1, output2) if output_size is int, else dict[int -> (out1, out2)].
    """
    if X_test is None or Y_test is None:
        X_test = X_train
        Y_test = Y_train

    dims = output_size if isinstance(output_size, list) else None

    # Separate model kwargs from training kwargs
    train_keys = {"num_anchors", "n_epochs"}
    model_kwargs = {k: v for k, v in kwargs.items() if k not in train_keys}
    train_kwargs = {k: v for k, v in kwargs.items() if k in train_keys}

    def _run_single(d):
        model = USCA(D=d, verbose=False, **model_kwargs)
        model.fit(X_train.cpu().numpy(), Y_train.cpu().numpy(), **train_kwargs)
        return model.transform(X_test.cpu().numpy(), Y_test.cpu().numpy())

    if dims is None:
        return _run_single(output_size)

    return {d: _run_single(d) for d in dims}


if __name__ == "__main__":
    view1 = torch.randn(100, 50)
    view2 = torch.randn(100, 50)
    output1, output2 = run_usca(view1, view2, 10, view1, view2)
    print("Output shapes:", output1.shape, output2.shape)

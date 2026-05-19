from baselines.jmds.JointMDS.joint_mds import JointMDS
import numpy as np


def run_jmds(X_train, Y_train, output_size, X_test=None, Y_test=None, **kwargs):
    """
    Run JointMDS.

    Args:
        X_train, Y_train: Training tensors.
        output_size (int or list[int]): Number of components. If a list,
            JMDS is fit once with max(output_size) and results are returned
            for each dim as a dict {dim: (out1, out2)}.
        X_test, Y_test: Test tensors (default to train if None).

    Returns:
        (output1, output2) if output_size is int, else dict[int -> (out1, out2)].
    """
    if X_test is None or Y_test is None:
        X_test = X_train
        Y_test = Y_train

    X_test = X_test[X_train.shape[0]:]
    Y_test = Y_test[Y_train.shape[0]:]

    dims = output_size if isinstance(output_size, list) else None
    fit_size = max(dims) if dims is not None else output_size

    print(X_test.shape, Y_test.shape)

    # shuffle Y_test
    perm = np.random.permutation(Y_test.shape[0])
    Y_test_shuffled = Y_test[perm]
    
    jmds = JointMDS(n_components=fit_size)
    out1_test, out2_test_shuffled, P = jmds.fit_transform(X_test.cpu().numpy(), Y_test_shuffled.cpu().numpy())
    out2_test = out2_test_shuffled[np.argsort(perm)]

    # This is what we would like to do but JMDS cannot generalize to unseen data
    # jmds.fit(X_train.cpu().numpy(), Y_train.cpu().numpy())
    # out1_full, out2_full = jmds.transform(X_test.cpu().numpy(), Y_test.cpu().numpy())

    X_full_aligned = torch.cat([torch.zeros((X_train.shape[0], out1_test.shape[1])).to(X_train), out1_test], dim=0)
    Y_full_aligned = torch.cat([torch.zeros((Y_train.shape[0], out2_test.shape[1])).to(Y_train), out2_test], dim=0)

    if dims is None:
        return X_full_aligned, Y_full_aligned

    return {d: (X_full_aligned[:, :d], Y_full_aligned[:, :d]) for d in dims}

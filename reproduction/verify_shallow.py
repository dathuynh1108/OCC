"""Independent dual-QP and source-extracted split/PCA equivalence fixtures."""

import ast
import types
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from .common import ROOT, verify_source, write_json
from .shallow import author, prepare, split_ids, svdd_diagnostics


def source_split(name, normal, seed, y, ty):
    path = ROOT / f"vendor/deep_svdd_theano/src/datasets/{name}.py"
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef))
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_data"
    )
    block = next(n for n in fn.body if isinstance(n, ast.If))
    # Exact source branch, with explicit Python 2 integer/range modernization only.
    code = ast.unparse(ast.Module(body=block.body, type_ignores=[]))
    code = code.replace("range(0, 10)", "list(range(0, 10))")
    code = code.replace(
        "self.n_train / Cfg.batch_size", "self.n_train // Cfg.batch_size"
    )
    cfg = types.SimpleNamespace(
        batch_size=200,
        out_frac=0,
        **{f"{name}_normal": normal, f"{name}_outlier": -1, f"{name}_val_frac": 0},
    )
    obj = types.SimpleNamespace(seed=seed)
    env = dict(
        np=np,
        Cfg=cfg,
        self=obj,
        X=np.arange(len(y))[:, None],
        y=y,
        X_test=np.arange(len(ty))[:, None],
        y_test=ty,
        extract_norm_and_out=author.extract_norm_and_out,
    )
    exec(compile(code, str(path), "exec"), env)
    # Extract the actual source holdout block, through its accepted permutation.
    svm = (ROOT / "vendor/deep_svdd_theano/src/svm.py").read_text()
    start = svm.index("                    n_val_set =")
    end = svm.index("                    self.data._X_test =", start)
    import textwrap

    env["self"] = types.SimpleNamespace(data=obj)
    exec(compile(textwrap.dedent(svm[start:end]), "svm.py:holdout", "exec"), env)
    order = env["perm"]
    test = obj._X_test.ravel()
    return (
        obj._X_train.ravel(),
        test[order[:1000]],
        test[order[1000:]],
        np.random.get_state(),
    )


def main():
    verify_source("deep_svdd_theano")
    rows = []
    y = np.repeat(np.arange(10), 601)
    ty = np.tile(np.arange(10), 1000)
    for name in ["mnist", "cifar10"]:
        for normal, seed in [(0, 1), (4, 5), (9, 10)]:
            expected = source_split(name, normal, seed, y, ty)
            actual = split_ids(y, ty, normal, seed)
            for a, b in zip(expected[:3], actual[:3]):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(expected[3][1], actual[3][1])
            assert expected[3][2:] == actual[3][2:]
            rows.append(
                {
                    "dataset": name,
                    "normal_class": normal,
                    "seed": seed,
                    "split_rng_equal": True,
                }
            )
    rng = np.random.RandomState(42)
    x = rng.randint(0, 256, (80, 1, 4, 4)).astype(np.float32)
    tx = rng.randint(0, 256, (20, 1, 4, 4)).astype(np.float32)
    train, test, *_ = prepare(x, tx, np.arange(60))
    a = x[:60].copy()
    b = tx.copy()
    empty = np.empty((0, 1, 4, 4), dtype=np.float32)
    author.normalize_data(a, empty, b, scale=np.float32(255))
    author.rescale_to_unit_interval(a, empty, b)
    old = author.PCA
    author.PCA = lambda **kw: PCA(**kw, svd_solver="full")
    expected, _, expected_test = author.pca(a, empty, b, 0.95)
    author.PCA = old
    np.testing.assert_array_equal(train, expected)
    np.testing.assert_array_equal(test, expected_test)
    # Independent Gaussian SVDD dual QP, full training set, versus libsvm.
    points = rng.normal(size=(40, 3))
    gamma = 0.2
    nu = 0.2
    n = len(points)
    kernel = np.exp(-gamma * cdist(points, points, "sqeuclidean"))
    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(2 * kernel),
        q=-np.diag(kernel),
        A=sparse.vstack([np.ones((1, n)), sparse.eye(n)], format="csc"),
        l=np.r_[1, np.zeros(n)],
        u=np.r_[1, np.full(n, 1 / (nu * n))],
        eps_abs=1e-10,
        eps_rel=1e-10,
        max_iter=100000,
        polishing=True,
        verbose=False,
    )
    result = solver.solve()
    assert result.info.status == "solved"
    model = OneClassSVM(gamma=gamma, nu=nu, tol=1e-9).fit(points)
    alpha = np.zeros(n)
    alpha[model.support_] = model.dual_coef_.ravel() / (nu * n)
    error = float(np.max(np.abs(alpha - result.x)))
    assert error < 1e-6, error
    diagnostics = svdd_diagnostics(model, points, gamma, nu)
    write_json(
        ROOT / "artifacts/reproduction/fixtures/shallow_parity.json",
        {
            "passed": True,
            "split_cases": rows,
            "PCA_source_max_abs_error": 0,
            "independent_osqp_alpha_max_abs_error": error,
            "diagnostics": diagnostics,
            "scope": "modern sklearn source-protocol adaptation; not MATLAB numeric reproduction",
        },
    )


if __name__ == "__main__":
    main()

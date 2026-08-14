# Models

A fitted model is just another item, with two differences that matter: loading
one can execute code, and the object is only meaningful if the code that
defined it is still importable. Everything on this page follows from those two
facts.

## The normal case

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)

folio.add_model(
    "classifier",
    clf,
    inputs=["train_features", "train_labels"],
    description="RF, 100 trees, trained on the March review set",
)
```

```python
clf = folio.get_model("classifier")
preds = clf.predict(X_test)
```

`add()` also recognizes estimator instances (scikit-learn `BaseEstimator`,
XGBoost, LightGBM, CatBoost) and routes them to the model handler, so
`folio.add("classifier", clf)` works. `add_model()` exists because it accepts
*anything* picklable, and pickling an arbitrary object should be a choice you
made on purpose rather than one the library made for you.

The catalog records the model class, the serialization format, the installed
scikit-learn version, a checksum, your description, and the lineage.
`describe()` shows it under **Models**.

## Store the context, not just the weights

A model without its parameters and its score is half an artifact. Keep them as
ordinary items — they are then searchable, described, and readable without
loading the model:

```python
folio.add("model_params", clf.get_params(), description="RF hyperparameters")
folio.add("test_score", float(clf.score(X_test, y_test)))
folio.add("predictions", preds_df, inputs=["classifier", "holdout"])
```

There is no separate hyperparameter API. `estimator.get_params()` already owns
that information; DataFolio just stores what you hand it.

## Two formats: joblib and skops

```python
folio.add_model("classifier", clf)                 # joblib (default) -> .joblib
folio.add_model("pipeline", pipe, custom=True)     # skops           -> .skops
```

| | joblib (default) | skops (`custom=True`) |
| --- | --- | --- |
| Mechanism | pickle | typed, inspectable serialization |
| Loading executes arbitrary code | **yes** | no — unknown types are refused |
| Requires the defining class importable | yes | yes |
| Speed / size | faster, smaller | slower, larger |
| Good for | your own models, trusted folios | models you will share, or load from a folio you did not write |

Use joblib by default. Use skops when the folio will cross a trust boundary.

## Loading and trust

**joblib models are pickle.** Loading one runs whatever code the file says to
run. Only load models from folios you trust — the same rule as any `.pkl` on
your disk.

**skops models refuse unknown types** unless you say otherwise:

```python
folio.get_model("pipeline")
# ValueError: skops file at …/models/pipeline--r2.skops contains non-standard
# types that are not trusted by default: ['mylib.Clipper']. If you trust the
# folio's author, load with trusted=True (e.g. folio.get_model(name, trusted=True)).

folio.get_model("pipeline", trusted=True)   # after reading that list
```

The error names exactly which types it does not recognize. That list is the
thing to review before opting in — it is the whole security benefit.

!!! warning "skops does not free you from the class definition"
    A custom transformer must still be **importable at load time**, in either
    format. Loading a skops pipeline whose `Clipper` class is not on the path
    fails with `AttributeError: module 'mylib' has no attribute 'Clipper'`.
    What skops buys you is *safety* (no arbitrary execution, an auditable type
    list), not portability across environments that lack your code.

    Practical consequence: put custom transformers in an installed package or a
    module that ships with the analysis — never in a notebook cell, where the
    class lands in `__main__` and can never be re-imported.

```python
# mylib.py — importable from anywhere the analysis runs
from sklearn.base import BaseEstimator, TransformerMixin

class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, lo=1.0):
        self.lo = lo
    def fit(self, X, y=None):
        self.lo_ = np.percentile(X, self.lo, axis=0)
        return self
    def transform(self, X):
        return np.clip(X, self.lo_, None)
```

Inheriting from `BaseEstimator`/`TransformerMixin` is what makes the class
introspectable — `get_params()`, `set_params()`, pipeline compatibility, and a
clean skops representation all come from it.

## Keeping several models

Names and descriptions do the work; there is no model registry to configure.

```python
for name, model in candidates.items():
    folio.add_model(f"models/{name}", model.fit(X, y),
                    inputs=["train"], description=f"{name} candidate")
    folio.add(f"scores/{name}", float(model.score(X_test, y_test)))

scores = {n: folio.get(f"scores/{n}") for n in candidates}
best = max(scores, key=scores.get)

folio.metadata["best_model"] = best
folio.describe("models/*")
```

When one of them becomes *the* model, either copy it to a stable name or
snapshot the folio so the choice is recoverable:

```python
folio.add_model("classifier", folio.get_model(f"models/{best}"),
                description=f"Selected: {best}", overwrite=True)
folio.create_snapshot("baseline", description=f"{best}, acc={scores[best]:.3f}")
```

## Models DataFolio does not handle for you

There is no PyTorch, TensorFlow, or Keras handler, and there will not be one:
each has its own checkpoint format with its own version rules, and wrapping
them would mean owning that complexity. Use the framework's own writer and
store the file:

```python
torch.save(model.state_dict(), "checkpoint.pt")
folio.add_file("checkpoint.pt", name="encoder_weights",
               description="Encoder state_dict, epoch 40")

path = folio.get("encoder_weights")          # a path back
model.load_state_dict(torch.load(path))
```

You keep the framework's guarantees; the folio keeps the description, the
lineage, and the catalog entry. For very large checkpoints that already live
somewhere else, `reference_table()` is for tables only — store the path as a
JSON item, or `add_file()` if you want the folio to own the bytes.

## Next

- **[Snapshots](snapshots.md)** — freezing "the model the paper used".
- **[Sharing a folio](sharing.md)** — including the trust question.
- **[What DataFolio is not](limits.md)**.

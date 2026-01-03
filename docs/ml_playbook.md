## ML Project Playbook (UV + MLflow + Train/Val/Test + Registry)

### 0) Setup (one-time per repo)

* Use `src/` layout so imports work cleanly.
* Create + sync venv with UV.
* Run `uv pip install -e .` so `nyc_taxi_mlops` is importable.
* Run MLflow server locally and point code to it:

  ```python
  mlflow.set_tracking_uri("http://127.0.0.1:5000")
  mlflow.set_experiment("nyc_taxi_mlops_experiment")
  ```

---

## 1) Data ingestion: pull raw data → produce processed dataset

**Goal:** one repeatable command that fetches data and writes a cleaned subset.

* `make_dataset.py`

  * download (or read) raw data from a URL (e.g., TLC parquet for specific `year/month`)
  * basic cleaning + column selection
  * write `data/processed/train.parquet`

**If you want a different month/year:** change CLI args or config in `make_dataset.py`.

---

## 2) Freeze the split: standard train/val/test (saved to disk)

**Goal:** ensure tuning and final training use the *same* split, always.

* `standard_data_split.py`

  * reads `data/processed/train.parquet`
  * writes:

    * `data/processed/splits/train.parquet`
    * `data/processed/splits/val.parquet`
    * `data/processed/splits/test.parquet`

**Why this matters:** prevents accidental leakage (e.g., “test == val”) and makes experiments comparable.

**If you want a time-based split later:** create another split script (e.g., `time_based_split.py`) that writes the same 3 output files. Then `tune.py` and `train.py` don’t change.

---

## 3) Feature code: one reusable `make_features(df)` function

**Goal:** keep feature logic reusable across tuning, training, and prediction.

* `features/build_features.py`

  * `make_features(df) -> (X, y, feature_cols)`
  * selects input columns and target column
  * does any feature engineering (datetime features, distance, etc.)
  * returns:

    * `X` (features)
    * `y` (label)
    * `feature_cols` (for logging/debugging)

**If you want different features:** only change `make_features()`.

---

## 4) Hyperparameter tuning: tune on train, score on val (never touch test)

**Goal:** explore model configs, log everything, pick a good “recipe”.

* `models/tune.py`

  * loads:

    * `splits/train.parquet`
    * `splits/val.parquet`
  * builds `X_train, y_train`, `X_val, y_val` using `make_features()`
  * runs random search over a defined search space
  * logs each trial to MLflow as a run (often nested under a parent run)

**What you typically log in MLflow per trial**

* parameters (hyperparameters)
* metrics (e.g., `rmse`, `mae` on val)
* tags:

  * `comparison_group` (helps filtering)
  * `split_strategy` (e.g., `standard_random_split_v1`)
  * `variant` (e.g., `xgb_trial`)

**How you choose the best model**

* Usually manual at first:

  * filter runs by `comparison_group`
  * sort by `rmse` (or `mae`)
  * pick the winner based on val metric + sanity

**If you tune a different model family (e.g., SVR):**

* add another branch in `tune.py`:

  * new `sample_params()`
  * new model constructor
  * same logging pattern
* keep everything else the same.

---

## 5) Final training: lock best params, train on train+val, evaluate on test

**Goal:** produce one “final” run whose test metric you trust.

* `models/train.py`

  * loads:

    * `splits/train.parquet`
    * `splits/val.parquet`
    * `splits/test.parquet`
  * combines train + val → fits final model
  * evaluates only once on test
  * logs:

    * `rmse_test`, `mae_test`
    * the trained model artifact
    * metadata (feature columns, split strategy, etc.)
  * includes a `BEST_*_PARAMS` dict that you paste from the chosen tuning run

**Important best-practice:**
Don’t keep “tuning” inside `train.py`. `train.py` should be deterministic and boring.

---

## 6) Register + pin the production model in MLflow Registry

**Goal:** one canonical model pointer you can always load.

* In MLflow UI:

  1. open the final training run
  2. register the model artifact under a name (e.g., `nyc_taxi_fare_model`)
  3. set alias `prod` to the version you want (MLflow 3.x style)

Now you have a stable reference:

* `models:/nyc_taxi_fare_model@prod`

---

## 7) Load production model via helper (for any downstream use)

**Goal:** code loads “prod” without knowing versions or file paths.

* `models/registry.py`

  * `load_production_model()` loads:

    ```python
    mlflow.pyfunc.load_model("models:/nyc_taxi_fare_model@prod")
    ```

This is the handoff point for anything that needs predictions.

---

# Flexibility: what to change when requirements change

### Change dataset / month / year

* modify `make_dataset.py` CLI args or config
* re-run ingestion + split + tune + train

### Change split strategy (random → time-based)

* add a new split script that writes the same `splits/train|val|test.parquet`
* re-run split + tune + train

### Change features

* change `make_features(df)`
* re-run tune + train
* (keep split constant if you want clean comparisons)

### Tune a new model type

* update `tune.py` to include a new branch (new search space + model constructor)
* update `train.py` to include the locked params + model constructor for that model

---

# Limitations of this approach (honest)

* Best params are “locked” by manual copy (good early; later can be automated).
* No cross-validation yet (fine for now; add later if you need robustness).
* Random split may be optimistic for time-dependent data (use time split for realism).
* Feature pipeline is code-based, not versioned as a standalone artifact (fine early; later you can log feature config / commit hashes / data hashes).

---

# Minimal “happy path” checklist

1. `make_dataset.py` → writes `data/processed/train.parquet`
2. `standard_data_split.py` → writes `data/processed/splits/{train,val,test}.parquet`
3. `tune.py` → logs many MLflow runs, choose best on val metric
4. `train.py` → trains on train+val, reports test metric, logs final model
5. register model + set alias `prod`
6. `registry.py` loads `models:/<name>@prod`

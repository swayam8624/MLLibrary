# From a CSV row to AutoML: a complete, source-linked C++23 learning lab

This laboratory is a deep reading path through the **classic-header C++23** tabular machine-learning stack in MLLibrary. It builds on [SOURCE_READING_ORDER.md](SOURCE_READING_ORDER.md) and [SOURCE_TEACHING_PROGRESS.md](SOURCE_TEACHING_PROGRESS.md). It does **not** require PyTorch, scikit-learn, KairoMath modules, or GPU execution.

The real call chain is:

```text
CSV file + explicit DatasetSchema
  -> Data/dataset.cpp::Dataset::from_csv
  -> typed Dataset rows + content fingerprint
  -> optional Data/profiling.cpp::profile_dataset
  -> Evaluation/cross_validation.cpp::CrossValidator::run
       -> fold-specific Dataset::select_rows
       -> Preprocessing/pipeline.cpp::FittedPipeline::fit_transform(TRAIN)
       -> FittedPipeline::transform(VALIDATION)   [no refitting]
       -> Experiment/experiment.cpp::ExperimentRunner::run
          -> make_estimator -> Classical/classical.cpp::fit
          -> predict -> Evaluation/metrics.cpp
  -> aggregate CV score per estimator
  -> Search/search.cpp::HyperparameterSearch (many candidate configurations)
  -> AutoML/automl.cpp::AutoMLPlanner + AutoMLRunner (many estimator families)
```

Every layer has different ownership and validation responsibilities. Knowing the types alone is not sufficient: you must know **when statistics are learned**, **which rows are visible to fitting**, and **what counts as held-out evidence**.

## 1. Two ingestion paths that must not be confused

[`Data/data.cpp`](../MLLibrary/Data/data.cpp) exposes the original MNIST entrypoint's `mat_load`, `one_hot_encode` and terminal `draw_mnist_digit`. `mat_load` expects a **raw binary Float32** file of exactly the caller-specified size, allocates its `Matrix` inside `MemArena`, and returns `nullptr` on failure. It does not parse the standard MNIST IDX file format and it does not perform byte-order conversion. `one_hot_encode` reads the first label value per row, casts it to an unsigned index, and silently ignores values outside the supported class range; input validation is a caller responsibility.

[`Data/dataset.cpp`](../MLLibrary/Data/dataset.cpp) instead stores independent **typed tabular rows** with `DatasetSchema`, `ColumnType`, and `DataValue = variant<monostate,double,bool,string>`. `Dataset::from_csv` parses quotes/newlines, checks header and column width, converts values against an explicit schema, and returns an owning `Dataset`. Numeric fields reject non-finite/malformed values, boolean fields accept documented spellings, and categorical/text fields remain strings until preprocessing. `DatasetError` carries an error code and, when applicable, physical CSV row/column context.

This distinction matters because the tabular evaluator cannot accept arbitrary raw MNIST matrix pointers as though they were an owned typed Dataset.

### A minimal typed CSV example

Suppose `toy.csv` contains exactly:

```csv
age,region,churn
22,east,0
37,west,1
29,east,0
49,west,1
31,north,0
42,north,1
26,west,0
51,east,1
```

This illustrative client code uses public declarations from [`dataset.hpp`](../MLLibrary/Data/dataset.hpp), [`pipeline.hpp`](../MLLibrary/Preprocessing/pipeline.hpp), and [`cross_validation.hpp`](../MLLibrary/Evaluation/cross_validation.hpp). It is a worked example, **not an executed build result**.

```cpp
// Build it as a client of MLLibraryCore using the repository's CMake
// include directories and link settings.
#include "dataset.hpp"
#include "pipeline.hpp"
#include "cross_validation.hpp"
#include "experiment.hpp"
#include <iostream>

int main() {
    using namespace mllibrary;

    // The target is intentionally marked, but is NOT a feature.
    data::DatasetSchema schema{{
        {"age", data::ColumnType::Numeric, false, false},
        {"region", data::ColumnType::Categorical, false, false},
        {"churn", data::ColumnType::Numeric, false, true}
    }};
    data::Dataset dataset = data::Dataset::from_csv("toy.csv", schema);

    // First study one explicit TRAIN-only fit and a frozen validation
    // transform. This illustrates what CrossValidator does per fold.
    auto split = dataset.split({
        .training_fraction = 0.75,
        .validation_fraction = 0.25,
        .test_fraction = 0.0,
        .seed = 5489,
        .shuffle = true
    });
    preprocessing::FittedPipeline pipeline;
    auto training = pipeline.fit_transform(split.training);
    auto validation = pipeline.transform(split.validation);
    std::cout << "Features per row: " << training.column_count
              << " / " << validation.column_count << '\n';

    // Now use a fold evaluator. With eight toy rows, two-fold
    // stratification is an educational choice, not credible performance
    // evidence. Real data needs substantially more samples.
    evaluation::CrossValidationConfig config;
    config.task = experiment::TaskKind::Classification;
    config.estimator.kind = experiment::EstimatorKind::LogisticRegression;
    config.strategy = evaluation::FoldStrategy::Stratified;
    config.primary_metric = evaluation::PrimaryMetric::MacroF1;
    config.folds = 2;
    config.seed = 5489;

    const auto result = evaluation::CrossValidator(config).run(dataset);
    if (!result.success) {
        std::cerr << "Evaluation error: " << result.error << '\n';
        return 1;
    }
    std::cout << "Mean MacroF1: " << result.aggregate.mean << '\n';
}
```

**Important:** this toy is designed for understanding shapes and fold boundaries. Its score is not trustworthy as an accuracy claim or a production model evaluation.

## 2. Type inference is useful but is not ground truth

[`profiling.cpp`](../MLLibrary/Data/profiling.cpp) can inspect a bounded sample of CSV rows to propose a `DatasetSchema`. It tracks whether nonmissing tokens parse as boolean or numeric, and for remaining text uses distinct-value counts/ratios and length thresholds to suggest categorical columns. `load_inferred_csv` then calls the actual typed loader.

A column containing numeric-looking postal codes or product identifiers might be inferred as numeric even though arithmetic would be meaningless. An integer-valued target might be a class ID instead of a regression target. A column's type is a modeling decision and must be checked against semantics.

`profile_dataset` reports counts, missingness, duplicate rows, frequency tables, numeric summaries, target majority fraction, warnings and a dataset fingerprint. This is descriptive data analysis. In the current `AutoMLPlanner`, profiling is performed over the Dataset supplied to the planner. If that is the entire project dataset, the planner has seen whole-dataset distributional facts **before** selecting candidate model families. This is not equivalent to fitting the estimator, but it is still selection context and must be acknowledged when interpreting a later score. Ideally keep an independent locked test partition outside model-family discovery.

The FNV-style dataset fingerprint is useful for detecting accidental changes to schema or records and for linking reports, but it is not a cryptographic hash or proof of trustworthy data provenance.

## 3. Why the train-only pipeline is the central leakage gate

[`FittedPipeline::fit`](../MLLibrary/Preprocessing/pipeline.cpp) learns numeric imputation values (mean, median or constant), numeric scales (none/standard/minmax), boolean missing-value handling, and sorted categorical vocabularies for one-hot features. The target is not included as an input feature. Text columns are dropped by default unless the caller requests a policy for them.

`transform` applies the existing state without modifying the learned means/categories. Unknown categories are either ignored (all-zero encoding for that column group) or rejected, according to `UnknownCategoryPolicy`. The transformed result is a **row-major double-valued feature matrix**, with separate typed target values.

Suppose a numeric feature has training values `[1,3,5]` and one validation value `100`. If you fit the scaler using both train and validation, its mean includes 100 and the training representation changes before the model is scored. That is data leakage. The correct model learns the mean from `[1,3,5]` alone and transforms 100 using those fixed statistics.

The same holds for one-hot categories: a category only seen in held-out rows must not silently expand a feature vocabulary learned from training. Otherwise the feature dimension and data-generating assumptions depend on evaluation observations.

`FittedPipeline::save/load` persist policy, fitted schema signature, feature names, training fingerprint and learned column state. This is a preprocessing artifact, **not** a complete trained-model checkpoint.

## 4. Classical models are algorithms, not simply names

[`Classical/classical.cpp`](../MLLibrary/Classical/classical.cpp) implements separate `fit` and `predict` methods over rectangular `DenseTable` rows:

- `StandardScaler` learns per-feature mean and standard deviation; near-constant columns use a scale of one.
- `PCA` centers training rows, builds a covariance matrix and estimates directions through power iteration and deflation. This is not a guarantee of exact SVD or arbitrary-rank numerical stability.
- `KMeans` starts from the first sample, chooses further farthest-first centers, then alternates assignment and mean updates. Clusters are unsupervised and labels are arbitrary.
- `LogisticRegression` accepts binary 0/1 labels and trains with batch gradient descent; `predict_probability` uses a sign-aware sigmoid.
- `LinearRegression` uses gradient descent for a linear numeric prediction. Its objective and regularization differ from classification.
- `KNearestNeighbors` stores the entire training table; prediction ranks distances and votes. Training is cheap, inference grows with dataset size.
- `GaussianNaiveBayes` learns class priors/feature-wise Gaussian means and smoothed variances, using log probabilities at inference.
- `DecisionTreeClassifier` finds recursive impurity-based splits; `RandomForestClassifier` trains seeded bootstrap trees and majority-votes predictions.

These are the classic implementations; [`ExperimentRunner`](../MLLibrary/Experiment/experiment.cpp) wraps them in a common interface and converts `PreparedDataset` double features into Float32 `DenseTable` for the relevant algorithms. Numerical comparison must therefore distinguish preprocessing precision from estimator precision.

**Worked question:** why can KNN be expensive at prediction time even with no gradient-based fitting? Because each query compares against stored rows. Why can a decision tree report an implausibly perfect training score? Because deep threshold splits can memorize training examples; independent held-out folds are needed.

## 5. Four cross-validation modes, four different assumptions

[`CrossValidator`](../MLLibrary/Evaluation/cross_validation.cpp) supports:

| Strategy | Which rows are withheld? | What it protects against |
| --- | --- | --- |
| KFold | Round-robin validation buckets, optionally after seeded shuffle | Basic out-of-sample evaluation |
| Stratified | Label buckets distributed across folds | Severe class imbalance in arbitrary splits, within available class counts |
| Group | Whole groups assigned to folds, larger groups placed first | The same entity/group leaking across train and validation |
| TimeSeries | Earlier rows train, later contiguous time blocks validate | Predicting the past using future observations |

`CrossValidator::run` selects fold indices, optionally removes the group/time control column from model features, takes `Dataset::select_rows` subsets, constructs a **fresh `FittedPipeline` per fold**, fits it on training only, transforms validation, trains a new estimator and computes that fold's metric. Only after all folds does it aggregate scores.

This is more than a CSV splitting feature: the correctly placed `fit_transform(training_raw)` and `transform(validation_raw)` calls are the actual leakage protection. Using one pipeline fitted on the original complete dataset for all folds would invalidate this property.

**Exercise:** design three tests. In group CV, every group ID must appear on one side of each fold but not both. In time CV, no training timestamp may be later than a validation timestamp. In a preprocessing test, an extreme validation-only feature value must not change the training-fold fitted mean.

## 6. Metrics mean different things

[`metrics.cpp`](../MLLibrary/Evaluation/metrics.cpp) builds a confusion matrix with rows as actual and columns as predicted classes, then precision, recall and F1 per class, macro averages and support-weighted F1. Accuracy summarizes correct class IDs; it can be misleading when almost every row belongs to the same class.

Binary log loss and ROC-AUC need positive-class probability scores, currently restricted to two classes in this helper. ROC-AUC is undefined for a fold with no positive or no negative examples; the implementation represents that condition with `std::nullopt`.

Regression metrics include MAE, MSE, RMSE, R² and a version of MAPE that excludes targets with near-zero denominator. A reported percentage error is not meaningful unless that exclusion rule is explained.

Scoring direction also changes: **higher is better** for MacroF1, Accuracy and R²; **lower is better** for RMSE and log loss. [`metric_is_maximized`](../MLLibrary/Search/search.cpp) controls search ranking accordingly.

## 7. ExperimentRunner versus Search versus AutoML

[`ExperimentRunner::run`](../MLLibrary/Experiment/experiment.cpp) accepts already prepared training/validation/test tables, creates a typed estimator via `make_estimator`, fits once, predicts on optional holdouts, records fit time and returns a result with errors and lineage identifiers.

[`HyperparameterSearch::run`](../MLLibrary/Search/search.cpp) instead evaluates **many configs** on CV folds. Grid search enumerates the Cartesian product; random search seeded-shuffles a discrete candidate list; successive halving begins with smaller fold budgets and promotes stronger candidates. The implementation bounds the search space and respects a wall-clock budget **between** trials. This budget is not a preemptive cancellation of an active model fit.

[`AutoMLPlanner::plan`](../MLLibrary/AutoML/automl.cpp) uses data profile and task heuristics to propose families. Current classification options include binary logistic regression, naive Bayes, decision trees, optional random forests and bounded KNN. The present regression route plans **linear regression only**, explicitly warning about absent regression trees/forests. [`AutoMLRunner::run`](../MLLibrary/AutoML/automl.cpp) divides its total trial/time budget among selected families, runs a search for each, and chooses a successful candidate based on the configured metric.

The internal interfaces do not guarantee that every available algorithm will be chosen for every dataset. For example, KNN may be skipped for large or high-width datasets because its exact inference cost would be excessive.

## 8. A selection score is not a fresh test score

There are three separate observations:

1. **Training performance:** computed on records used to fit weights or split thresholds.
2. **Cross-validation performance:** computed on folds that were held out from that fold's fitting. This is useful to compare configurations.
3. **Locked final test performance:** computed only after candidate family, hyperparameters, preprocessing and thresholds have been frozen.

Trying hundreds of models on the same CV folds can overfit the *selection process* to those folds, even if every individual fit obeys correct train-only preprocessing. A mean CV score selected from hundreds of noisy trials is optimistically biased relative to a new independent evaluation.

The current AutoML winner comes from cross-validation selection; it does **not** automatically represent independent final-test performance. If the user needs a publishable benchmark, reserve a test partition at the start and do not pass it to the planner/search loop; or use nested CV for an outer unbiased estimate.

## 9. Reproducibility and reporting checklist

A defensible experiment report records dataset source, typed schema, fingerprint, exact split indices or seed/strategy, preprocessing policy and fitted state, estimator hyperparameters, each CV fold's metrics, aggregate mean/spread, failed/pruned trials, compiler/platform information, runtime timing methodology and final untouched-test evaluation.

The source's JSON/HTML reporting helpers serialize selected evidence, not every hidden influence on a result. In particular, FNV fingerprints are not integrity signatures, a fixed random seed alone is not a complete reproducibility manifest, and host wall-clock fit time is not a portable performance guarantee.

### Exercises tied to source and tests

1. **CSV parser correctness:** quotes, escaped quotes, CRLF, empty quoted versus unquoted field, malformed numerics and schema-header disagreement.
2. **Split reproducibility:** same seed produces the same chosen row indices; different splits have no overlapping indices when generated by the provided split builder.
3. **Fold leakage:** assert train/validation disjointness for KFold; group exclusivity for Group; future-only validation for TimeSeries.
4. **Pipeline vocabulary:** categorical level present only in validation must not change feature width; test both `Ignore` and `Error`.
5. **Metric edge cases:** one-class ROC-AUC, no positive predictions, constant regression targets, zero-valued MAPE denominator.
6. **Estimator parity:** fit each classical family on a small hand-checkable dataset, then test expected predictions and invalid feature widths.
7. **Selection protocol:** compare grid/random/halving trial histories and final winner, including failed trials; never evaluate the final test partition inside candidate selection.
8. **Reporting:** inspect emitted JSON and dataset fingerprints, check file errors, and retain all unsuccessful-trial diagnostic records.

**Verification status:** documentation based on the checked-in C++23 source and public headers. The example has not been compiled or executed during this teaching pass; no measured model score, native build success or external-data claim is implied.

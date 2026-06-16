# Getting Started with Stamp

This guide is designed to help you get started with **multiplex pathology data**
in STAMP. The focus here is channel-ordered `.qptiff`, `.tiff`, and `.tif`
slides, multiplex feature extraction, and downstream marker-aware modeling.

To follow along, you will need:

- multiplex slide files
- the marker order for the slide channels
- a patient-level clinical table
- a slide table mapping each patient to a feature file
- a target label for training or cross-validation

The main workflow in this guide is:

`multiplex slides -> multiplex .h5 features -> crossval / training / statistics`

> [!NOTE]
> If you prefer a browser-based workflow, see [`STAMP-Workbench`](https://github.com/KatherLab/STAMP-Workbench), a separate web UI for building and monitoring STAMP pipelines. You can install it into the same environment with `uv pip install git+https://github.com/KatherLab/STAMP-Workbench` and launch it from your STAMP checkout with `stamp-workbench`.

## Multiplex Slide Inputs

For the next steps we assume that all multiplex slides are stored under the same
root directory. We will call this the _slide directory_. STAMP discovers slides
recursively under that directory.

Supported multiplex inputs for this path are:

- `.qptiff`
- `.tiff`
- `.tif`

Each image channel must correspond to one marker, and the channel order in the
file must match the order you define in `preprocessing.markers`.

> [!IMPORTANT]
> STAMP does not infer marker names from the image file. The `markers` list in
> the config is the source of truth for channel semantics.

Brightfield WSIs are still supported and must be in a format [supported by OpenSlide][openslide].

[openslide]: https://openslide.org/#about-openslide "About OpenSlide"

## Creating a Configuration File

Stamp is configured using configuration files.
We recommend creating one configuration file per experiment
and storing in the same folder as the eventual results,
as this makes it easier to reconstruct which data and parameters a model was trained with later.

The `stamp init` command creates a new configuration file with dummy values.
By default, it is created in `$PWD/config.yaml`,
but we can use the `--config` option to specify its location:
```sh
# Create a directory to save our experiment results to
mkdir stamp-test-experiment
# Create a new config file in said directory
stamp --config stamp-test-experiment/config.yaml init
```

## Multiplex Feature Extraction

Before training, we first convert multiplex slides into feature files that are
easy to reuse across experiments. The example below uses [KRONOS][kronos] as
one multiplex-capable extractor because it preserves both patch-level and
per-marker signal.

STAMP still supports the broader extractor set for preprocessing workflows:

- [ctranspath][ctranspath]
- [chief_ctranspath][chief_ctranspath]
- [DinoBloom][dinobloom]
- [CONCH][conch]
- [CONCHv1.5][conch1_5]
- [UNI][uni]
- [UNI2][uni2]
- [Virchow][virchow]
- [Virchow2][virchow2]
- [Gigapath][gigapath]
- [H-optimus-0][h_optimus_0]
- [H-optimus-1][h_optimus_1]
- [mSTAR][mstar]
- [MUSK][musk]
- [PLIP][plip]
- [KEEP][keep]
- [TICON][ticon]
- [RedDino][reddino]
- [KRONOS][kronos]

Please refer to the [installation instructions](README.md#installation) to make
sure the extractor dependencies are available in your environment.

Open the `stamp-test-experiment/config.yaml` we created in the last step and
update the `preprocessing` section with absolute paths and multiplex-specific
settings.

```yaml
# stamp-test-experiment/config.yaml

preprocessing:
  output_dir: "/absolute/path/to/stamp-test-experiment"
  wsi_dir: "/absolute/path/to/multiplex_slide_dir"
  mode: "multiplex"
  extractor: "kronos"
  # Use CUDA for multiplex preprocessing. With parallel: true,
  # STAMP splits slides across all visible GPUs.
  device: "cuda"

  # Optional: split slides across all visible GPUs.
  parallel: true

  # KRONOS expects tile_size_px divisible by 16.
  tile_size_px: 256
  tile_size_um: 256.0

  # Optional cache for repeated extraction experiments.
  cache_dir: "/absolute/path/to/stamp-test-experiment/../cache"

  # Optional CSV with columns: marker_name, marker_mean, marker_std
  # marker_metadata_csv: "/absolute/path/to/marker_metadata.csv"

  # Markers must be listed in the same order as the image channels.
  markers:
    - name: "DAPI"
      # Optional per-marker normalization values used during preprocessing.
      # If omitted, STAMP tries to auto-fill them from bundled metadata
      # or from marker_metadata_csv when available.
      # mean: 0.0832
      # std: 0.0959
    - name: "CD8"
    - name: "Ki67"
    - name: "GrzB"
    - name: "FOXP3"
    - name: "CD4"
    - name: "CD3"

  # Keep this at 1 for multiplex preprocessing.
  max_workers: 1
```

For LZW-compressed multiplex TIFF/QPTIFF files, STAMP requires `imagecodecs`:

```sh
uv pip install --python .venv/bin/python imagecodecs
```

The `mean` and `std` fields are optional per marker, but they matter because
STAMP uses them for channel normalization. You can provide them inline under
each marker entry, or let STAMP fill missing values from `marker_metadata_csv`
or from its bundled multiplex marker metadata when the marker names match.

For **multi-channel multiplex preprocessing**, keep `max_workers: 1`. In this
workflow, scaling comes mainly from the model/device path and optional
`parallel: true` multi-GPU slide splitting rather than larger worker counts.

`parallel: true` only affects CUDA runs. If `device: "cpu"` is used, STAMP does
not launch multi-GPU preprocessing workers.

Extract the features with:

```sh
stamp --config stamp-test-experiment/config.yaml preprocess
```
Depending on cohort size and hardware, this process may take hours to days.

You can interrupt this process and resume later by running the same command again.

This writes one `.h5` file per slide with:

- `feats`
- `patch_embeddings`
- `marker_embeddings`
- `token_embeddings`
- `coord_x`
- `coord_y`

as well as HDF5 attributes such as:

- `marker_names`
- `marker_means`
- `marker_stds`

This layout matches the multiplex feature structure used in
`multiplex-notebooks_qtif`, so notebook exploration can move into STAMP runs
without reshaping the data format.

If a multiplex slide contains more channels than listed in `markers`, STAMP uses
the first `len(markers)` channels in file order and logs a warning. If a slide
contains fewer channels than configured, preprocessing fails for that slide.

> In case you want to use a gated model such as KRONOS, you may need to log in:
> ```
>huggingface-cli login
> ```
> More info about this [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

> **If you are using the UNI or CONCH models**
> and working in an environment where your home directory storage is limited,
> you may want to also specify your huggingface storage directory
> by setting the `HF_HOME` environment variable:
> ```sh
> export HF_HOME=/path/to/directory/to/store/huggingface/data/in
> huggingface-cli login   # only needs to be done once per $HF_HOME
> stamp -c stamp-test-experiment/config.yaml preprocess
> ```

[ctranspath]: https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043 "Transformer-based unsupervised contrastive learning for histopathological image classification"
[dinobloom]: https://github.com/marrlab/DinoBloom "DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology"
[uni]: https://www.nature.com/articles/s41591-024-02857-3 "Towards a general-purpose foundation model for computational pathology"
[uni2]: https://huggingface.co/MahmoodLab/UNI2-h
[conch]: https://www.nature.com/articles/s41591-024-02856-4 "A visual-language foundation model for computational pathology"
[conch1_5]: https://huggingface.co/MahmoodLab/conchv1_5
[virchow]: https://huggingface.co/paige-ai/Virchow "A foundation model for clinical-grade computational pathology and rare cancers detection"
[virchow2]: https://huggingface.co/paige-ai/Virchow2
[chief_ctranspath]: https://github.com/hms-dbmi/CHIEF
[gigapath]: https://huggingface.co/prov-gigapath/prov-gigapath
[h_optimus_0]: https://huggingface.co/bioptimus/H-optimus-0
[h_optimus_1]: https://huggingface.co/bioptimus/H-optimus-1
[mstar]: https://huggingface.co/Wangyh/mSTAR
[musk]: https://huggingface.co/xiangjx/musk
[plip]: https://github.com/PathologyFoundation/plip
[keep]: https://loiesun.github.io/keep/ "A Knowledge-enhanced Pathology Vision-language Foundation Model for Cancer Diagnosis"
[TITAN]: https://huggingface.co/MahmoodLab/TITAN
[COBRA2]: https://huggingface.co/KatherLab/COBRA
[EAGLE]: https://github.com/KatherLab/EAGLE
[MADELEINE]: https://huggingface.co/MahmoodLab/madeleine
[PRISM]: https://huggingface.co/paige-ai/Prism
[TICON]: https://cvlab-stonybrook.github.io/TICON/ "TICON: A Slide-Level Tile Contextualizer for Histopathology Representation Learning"
[reddino]: https://github.com/Snarci/RedDino "RedDino: A Foundation Model for Red Blood Cell Analysis"
[kronos]: https://huggingface.co/MahmoodLab/kronos


## Doing Cross-Validation on the Data Set

Once multiplex features have been extracted, the usual next step is
cross-validation. This lets you test whether the feature representation carries
signal for your target before setting up a separate deployment workflow.

To perform a cross-validation, add the following lines to your
`stamp-test-experiment/config.yaml`. Set `feature_dir` to the directory that now
contains your multiplex `.h5` files. `clini_table` and `slide_table` should
point to Excel or CSV files with the required patient and filename columns.

`ground_truth_label` is the clinical target you want to predict. For
single-target classification, use one column name. For multi-target
classification, use a list of column names and set
`advanced_config.model_name: "barspoon"`. For single-target runs, we recommend
explicitly setting `categories`.

```yaml
# stamp-test-experiment/config.yaml

crossval:
  output_dir: "/absolute/path/to/stamp-test-experiment"

  # An excel (.xlsx) or CSV (.csv) table containing the clinical information of
  # patients.  Patients not present in this file will be ignored during training.
  # Has to contain at least two columns, one titled "PATIENT", containing a patient ID,
  # and a second column containing the categorical ground truths for that patient.
  clini_table: "metadata-CRC/TCGA-CRC-DX_CLINI.xlsx"

  # Directory the extracted features are saved in.
  feature_dir: "/absolute/path/to/stamp-test-experiment/multiplex_features"

  # A table (.xlsx or .csv) relating every patient to their feature files.
  # The table must contain at least two columns, one titled "PATIENT",
  # containing the patient ID (matching those in the `clini_table`), and one
  # called "FILENAME", containing the feature file path relative to `feature_dir`.
  # Patient IDs not present in the clini table as well as non-existent feature
  # paths are ignored.
  slide_table: "slide.csv"

  # Name of the column from the clini table to train on.
  ground_truth_label: "isMSIH"
  # For multi-target classification with barspoon:
  # ground_truth_label: ["subtype", "grade"]

  # Optional settings:

  # The categories occurring in the target label column of the clini table.
  # If unspecified, they will be inferred from the table itself.
  categories: ["yes", "no"]
  # For multi-target classification, per-target categories are inferred.

  # Number of folds to split the data into for cross-validation
  #n_splits: 5
```

After specifying all the parameters of our cross-validation,
we can run it by invoking:
```sh
stamp --config stamp-test-experiment/config.yaml crossval
```

### Multiplex Marker Selection

For multiplex `.h5` feature files, STAMP can train marker-aware models directly
from per-marker embeddings. For multiplex training, use
`advanced_config.model_name: "marker_fusion"`.
If your files contain:

- a dataset `marker_embeddings` with shape `(tiles, markers, features)`, and
- an HDF5 root attribute `marker_names`,

you can train `marker_fusion` on either all markers or only a selected subset.

For example, if your `.h5` files contain:

```text
marker_names = ["DAPI", "cMET", "PanCK", "DLL3", "Prame", "HER2", "Trop2"]
```

you can configure cross-validation like this:

```yaml
crossval:
  output_dir: "/absolute/path/to/stamp-test-experiment"
  clini_table: "/absolute/path/to/clini.csv"
  feature_dir: "/absolute/path/to/multiplex/features"
  slide_table: "/absolute/path/to/slide.csv"
  ground_truth_label: "KRAS"

  # Recommended for multiplex marker-aware training
  feature_dataset_name: "marker_embeddings"

  # Train on one marker:
  # selected_markers: "PanCK"

  # Or train on a subset of markers:
  selected_markers: ["PanCK", "HER2"]

advanced_config:
  # Use marker_fusion for multiplex marker-aware training
  model_name: "marker_fusion"
```

Notes:

- Multiplex training should use `model_name: "marker_fusion"`.
- `selected_markers` may be a single marker name or a list of marker names.
- Marker names are matched against the HDF5 `marker_names` attribute.
- If `selected_markers` is used with `model_name: "marker_fusion"` and
  `feature_dataset_name: "auto"`, STAMP will automatically load
  `marker_embeddings`.
- Deployment reuses the marker subset saved in the checkpoint, so inference stays
  consistent with training.
- Explainability also reuses the marker subset saved in the checkpoint. Marker
  labels are read from the `.h5` `marker_names` attribute by default, then
  aligned to the checkpoint subset order.


## Generating Statistics

After training and validating your model, you may want to generate statistics to evaluate its performance.
This can be done by adding a `statistics` section to your `stamp-test-experiment/config.yaml` file.
The configuration should look like this:

```yaml
# stamp-test-experiment/config.yaml

statistics:
  output_dir: "/absolute/path/to/stamp-test-experiment/statistics"

  # Name of the target label.
  ground_truth_label: "isMSIH"

  # A lot of the statistics are computed "one-vs-all", i.e. there needs to be
  # a positive class to calculate the statistics for.
  true_class: "yes"

  pred_csvs:
  - "/absolute/path/to/stamp-test-experiment/split-0/patient-preds.csv"
  - "/absolute/path/to/stamp-test-experiment/split-1/patient-preds.csv"
  - "/absolute/path/to/stamp-test-experiment/split-2/patient-preds.csv"
  - "/absolute/path/to/stamp-test-experiment/split-3/patient-preds.csv"
  - "/absolute/path/to/stamp-test-experiment/split-4/patient-preds.csv"
```

To generate the statistics, run the following command:
```sh
stamp --config stamp-test-experiment/config.yaml statistics
```

Afterwards, the `output_dir` should contain the following files:
  - `isMSIH-categorical-stats-individual.csv` contains statistical scores
    for each individual split.
  - `isMSIH-categorical-stats-aggregated.csv` contains the mean
    as well as the 95% confidence interval for the statistical scores
    for the splits.
  - `roc-curve_isMSIH=yes.svg` and `pr-curve_isMSIH=yes.svg`
    contain the ROC and precision recall curves of the splits.

## Slide-Level Encoding 
Tile-Level features can be enconded into a single feature per slide, this is useful
when trying to capture global patterns across whole slides.

STAMP currently supports the following encoders:
- [CHIEF][CHIEF_CTRANSPATH]
- [TITAN]
- [GIGAPATH]
- [COBRA2]
- [EAGLE]
- [MADELEINE]
- [PRISM]

Slide encoders take as input the already extracted tile-level features in the 
preprocessing step. Each encoder accepts only certain extractors and most
work only on CUDA devices:

| Encoder | Required Extractor | Compatible Devices | Notes
|--|--|--|--|
| CHIEF | CHIEF-CTRANSPATH | CUDA only | Text encoding removed
| TITAN | CONCH1.5 | CUDA, cpu, mps | 
| GIGAPATH | GIGAPATH | CUDA only
| COBRA2 | CONCH, UNI, VIRCHOW2 or H-OPTIMUS-0 | CUDA only
| EAGLE | CTRANSPATH, CHIEF-CTRANSPATH | CUDA only
| MADELEINE | CONCH | CUDA only
| PRISM | VIRCHOW_FULL | CUDA only

> **Note:** Slide-level features cannot be used directly for modeling because the clinical labels are at the patient level. However, if only one slide is available per patient, using **[Patient-Level Encoding](#patient-level-encoding)** will produce the same representation as slide-level encoding—but supports downstream modeling.

As with feature extractors, most of these models require you to request
access. The following example uses CHIEF, which is available if you installed 
STAMP with `uv sync --all-extras`. The configuration should look like this:

```yaml
# stamp-test-experiment/config.yaml

slide_encoding:
  # Encoder to use for slide encoding. Possible options are "cobra",
  # "eagle", "titan", "gigapath", "chief", "prism", "madeleine".
  encoder: "chief"
  
  # Directory to save the output files.
  output_dir: "/path/to/save/files/to"
  
  # Directory where the extracted features are stored.
  feat_dir: "/path/your/extracted/features/are/stored/in"
  
  # Device to run slide encoding on ("cpu", "cuda", "cuda:0", etc.)
  device: "cuda"

  # Optional settings:
  # Directory where the aggregated features are stored. Needed for
  # some encoders such as eagle (it requires virchow2 features).
  #agg_feat_dir: "/path/your/aggregated/features/are/stored/in"

  # Add a hash of the entire preprocessing codebase in the feature folder name.
  #generate_hash: True
  ```

Don't forget to put in `feat_dir` a path containing, in this case, `ctranspath` or
`chief-ctranspath` tile-level features. Once everything is set, you can simply run:

```sh
stamp --config stamp-test-experiment/config.yaml encode_slides
```
The output will be one `.h5` file per slide. 

## Patient-Level Encoding
Even though the available encoders are designed for slide-level use, this
option concatenates the slides of a patient along the x-axis, creating a single
"virtual" slide that contains two blocks of tissue. The configuration is the same
except for `slide_table` which is required to link slides with patients.
```yaml
# stamp-test-experiment/config.yaml

patient_encoding:
  # Encoder to use for patient encoding. Possible options are "cobra",
  # "eagle", "titan", "gigapath", "chief", "prism", "madeleine".
  encoder: "eagle"
  
  # Directory to save the output files.
  output_dir: "/path/to/save/files/to"
  
  # Directory where the extracted features are stored.
  feat_dir: "/path/your/extracted/features/are/stored/in"
  
  # A table (.xlsx or .csv) relating every slide to their feature files.
  # The table must contain at least two columns, one titled "SLIDE",
  # containing the slide ID, and one called "FILENAME", containing the feature file path relative to `feat_dir`.
  slide_table: "/path/of/slide.csv"
  
  # Device to run slide encoding on ("cpu", "cuda", "cuda:0", etc.)
  device: "cuda"

  # Optional settings:
  patient_label: "PATIENT"
  filename_label: "FILENAME"
  
  # Directory where the aggregated features are stored. Needed for
  # some encoders such as eagle (it requires virchow2 features).
  #agg_feat_dir: "/path/your/aggregated/features/are/stored/in"

  # Add a hash of the entire preprocessing codebase in the feature folder name.
  #generate_hash: True
  ```

  Then run:
  ```sh
stamp --config stamp-test-experiment/config.yaml encode_patients
```

The output `.h5` features will have the patient's id as name. 

## Training with Patient-Level Features

Once you have patient-level features, 
you can train models directly on these features. This is useful because:
- **Efficient with Limited Data**: Patient-level modeling often performs better when data is scarce, since pretrained encoders can extract robust features from each slide as a whole.
- **Faster Training & Reduced Overfitting**: With fewer parameters to train compared to tile-level models, patient-level models train more quickly and are less prone to overfitting.
- **Enables Interpretable Cohort Analysis**: Patient-level features can be used for unsupervised analyses, such as clustering, making it easier to interpret and explore patient subgroups within your cohort.

To train a model using patient-level features, you can use the same command as before:
```sh
stamp --config stamp-test-experiment/config.yaml crossval
```

The key differences for patient-level modeling are:
- The `feature_dir` should contain patient-level `.h5` files (one per patient).
- The `slide_table` is not needed since there's a direct mapping from patient ID to feature file.
- STAMP will automatically detect that these are patient-level features and use a MultiLayer Perceptron (MLP) classifier instead of the Vision Transformer.

You can then run statistics as done with tile-level features.

## Heatmaps and Top Tiles

The `stamp heatmaps` command generates visualization outputs to help interpret model predictions and identify which regions of the slide contribute most to the classification decision. This command creates:

- **Attention heatmaps**: Show which tiles the model focuses on for each class
- **Overlay visualizations**: Combine heatmaps with slide thumbnails for better spatial context
- **Class maps**: Display which class each tile is most associated with
- **Top/bottom tiles**: Extract the most and least predictive image patches from the predicted class. 

To generate heatmaps, you need a trained model checkpoint from either the train or crossval commands. The configuration file should look like this:

```yaml
# stamp-test-experiment/config.yaml

heatmaps:
  output_dir: "/absolute/path/to/stamp-test-experiment/heatmaps"

  # Directory where the extracted tile-level features are stored
  feature_dir: "/absolute/path/to/stamp-test-experiment/xiyuewang-ctranspath-7c998680-112fc79c"

  # Directory containing the original whole slide images
  wsi_dir: "/absolute/path/to/wsi_dir"

  # Path to the trained model checkpoint
  checkpoint_path: "/absolute/path/to/stamp-test-experiment/split-0/checkpoints/epoch=15-step=123.ckpt"

  # Optional settings:

  # Overlay plot opacity (0 = transparent, 1 = opaque)
  opacity: 0.6

  # Number of top-scoring tiles to extract for each slide
  topk: 5

  # Number of bottom-scoring tiles to extract for each slide  
  bottomk: 5

  # Specific slides to process (relative to wsi_dir)
  # If not specified, all slides in wsi_dir will be processed
  slide_paths:
  - slide1.svs
  - slide2.mrxs

  # Device to run heatmap generation on
  device: "cuda"
  ```

  > **Note:** Heatmaps currently only work with tile-level features. If you have slide-level or patient-level features, you'll need to use the original tile-level features for heatmap generation.

  Generate the heatmaps by running:

  ```sh
  stamp --config stamp-test-experiment/config.yaml heatmaps
  ```

  The heatmap command creates an organized folder structure for each slide:

  ```sh
  heatmaps/
└── slide-name/
    ├── plots/
    │   ├── overview-slide-name.png     # Complete overview with all classes
    │   └── overlay-slide-name-class.png # Individual class overlays
    ├── raw/             # Raw data files
    │   ├── thumbnail-slide-name.png         # Slide thumbnail
    │   ├── classmap-slide-name.png          # Class assignment map
    │   ├── slide-name-class=score.png       # Raw heatmap per class
    │   └── raw-overlay-slide-name-class.png # Overlay without legends
    └── tiles/           # Individual tile extractions
        ├── top_01-slide-name-class=score.jpg    # Highest scoring tiles
        ├── top_02-slide-name-class=score.jpg
        └── bottom_01-slide-name-class=score.jpg # Lowest scoring tiles
  ```

## Explainability for Multiplex Marker Models

The `stamp explainability` command generates marker-level outputs for multiplex
feature files. This is currently supported only for the `marker_fusion` model,
where each `.h5` file contains per-marker embeddings in the
`marker_embeddings` dataset.

For explainability, STAMP expects multiplex `.h5` files to contain:

- a dataset `marker_embeddings` with shape `(tiles, markers, features)`, and
- an HDF5 root attribute `marker_names`

The configuration can look like this:

```yaml
# stamp-test-experiment/config.yaml

explainability:
  output_dir: "/absolute/path/to/stamp-test-experiment/explainability"
  feature_dir: "/absolute/path/to/multiplex/features"
  checkpoint_path: "/absolute/path/to/stamp-test-experiment/model.ckpt"

  # Optional: choose which HDF5 dataset to read
  feature_dataset_name: "marker_embeddings"

  # Optional: only explain a subset of feature files
  #feature_paths:
  #- "slide_a.h5"
  #- "slide_b.h5"

  # Optional: for classification, explain this class index.
  # If omitted, STAMP explains the predicted class for each slide.
  #class_index: 1

  # Optional manual label override. Usually not needed.
  #marker_names: ["DAPI", "PanCK", "HER2"]

  export_tile_saliency: true
  device: "cuda"
```

Run:

```sh
stamp --config stamp-test-experiment/config.yaml explainability
```

Behavior notes:

- Multiplex explainability currently supports `marker_fusion` checkpoints only.
- If `marker_names` is omitted in the config, STAMP reads marker labels from
  the `.h5` `marker_names` attribute.
- If the checkpoint was trained with `selected_markers`, explainability uses
  that saved marker subset automatically and keeps the checkpoint order.
- Because of that, you usually do not need to repeat marker names or marker
  subsets in the explainability config.
- If a subset-trained checkpoint is used, the `.h5` files still need
  `marker_names` so STAMP can map the checkpoint subset back to the stored
  markers.


## Advanced configuration

Advanced experiment settings can be specified under the `advanced_config` section in your configuration file.
This section lets you control global training parameters, model type, and the target task (classification, regression, or survival).

```yaml
# stamp-test-experiment/config.yaml

advanced_config:
  seed: 42
  task: "classification" # or regression/survival
  max_epochs: 32
  patience: 16
  batch_size: 64
  # Only for tile-level training. Reducing its amount could affect
  # model performance. Reduces memory consumption. Default value works
  # fine for most cases.
  bag_size: 512
  #num_workers: 16 # Default chosen by cpu cores
  # One Cycle Learning Rate Scheduler parameters. Check docs for more info.
  # Determines the initial learning rate via initial_lr = max_lr/div_factor
  max_lr: 1e-4
  div_factor: 25. 
  # Select a model regardless of task
  # Available models are: vit, trans_mil, mlp, linear, barspoon
  model_name: "vit"

  model_params:
    vit: # Vision Transformer
      dim_model: 512
      dim_feedforward: 512
      n_heads: 8
      n_layers: 2
      dropout: 0.25
      use_alibi: false
```

> [!NOTE]
> STAMP automatically adapts its **model architecture**, **loss function**, and **evaluation metrics** based on the task specified in the configuration file.
>
> - **Regression**: requires only `ground_truth_label`.
> - **Survival analysis**: requires `time_label` (follow-up time) and `status_label` (event indicator).
> - **Multi-target classification**: requires `ground_truth_label` as a list and `advanced_config.model_name: "barspoon"`.
>
> These requirements apply consistently across cross-validation, training, deployment, and statistics.


Lexically-Scaled MaxEnt learning on Korean Irregular Verb/Adjective Conjugation
========================================================================================

This repository contains a modified implementation of the original Lexically-Scaled MaxEnt model(Hughto et al., 2019), updated to allow different priors on distinct morphological constituents. It also includes training-data generation code and a toy dataset modeled after Korean regular and irregular verb/adjective conjugation.

* The code runs on Python 3.  
  `train.py` takes largely the same arguments as the original model, except that the user must now manually specify three separate lambda values:

    - LAMBDA1  
    - LAMBDA2 (for the stem or preceding morphological constituent)  
    - LAMBDA3 (for the suffix or succeeding morphological constituent)

  **Usage:**
  
    ```bash
    python3 train.py \
        METHOD NEG_WEIGHTS NEG_SCALES \
        LAMBDA1 LAMBDA2 LAMBDA3 \
        L2_PRIOR INIT_WEIGHT RAND_WEIGHTS \
        ETA EPOCHS TD_FILE LANGUAGE OUTPUT_DIR
    ```

* Training output file names now explicitly indicate the epoch, learning rate, and all three priors applied during generation, making it easier to track experimental settings.

The generated data are stored in the `generate_data` folder.  
The folder also contains the code used to generate each suffix-count condition, with matching names:

- `6suffix.py` generates input data file with six suffix variants (`6suffix.tsv`)  
- `8suffix.py` generates input data file with eight suffix variants (`8suffix.tsv`)  
- … (and so on for other conditions)


Trained results are organized in subfolders within the output directory.

- For experiments with different prior settings on stems and suffixes, results are stored in the `differ_scales/` subfolder.
- For each suffix-count condition, results are stored in separate subfolders named by condition (e.g., `2_suffix/`, `4_suffix/`, …).

### Motivation for Manipulating Suffix Number and Priors

In Hughto et al. (2019), the scaled MaxEnt model performed well for Russian prefixation, where a few lexically indexed prefixes combine with many stems. However, the model failed to learn appropriate scale weights when the locus of lexical indexation shifts to stems that combine with only a few suffix types. In this configuration, the triggering stems fail to properly penalize relevant constraints, and the explanatory burden is incorrectly shifted onto the phonologically regular suffixes.

Two possible ways to address this issue are:

- **Increasing the number of suffixes** available for each stem, allowing it to develop more robust generalizations.

- **Manipulating priors**, by imposing a stronger regularization penalty on suffixes to prevent them from dominating learning, while decreasing the prior for stems to make them more flexible for acquiring appropriate weights.


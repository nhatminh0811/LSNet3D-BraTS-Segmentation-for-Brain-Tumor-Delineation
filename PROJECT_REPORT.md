# LSNet3D BraTS Segmentation for Brain Tumor Delineation

**Name:** `<Your Name>`  
**Student ID:** `<Your Student ID>`  
**Module:** UXCFXK-30-3 Digital Systems Project  
**Resources:** `<GitHub link>`, `<OneDrive link (dataset/checkpoints/report assets)>`

## Abstract
This project develops a 3D medical image segmentation system for automatic brain tumor delineation on BraTS MRI volumes. The proposed pipeline combines an LSNet3D-inspired encoder, a CBAM-enhanced 3D decoder, and deep supervision with multi-scale auxiliary outputs. The training objective uses a combined weighted Cross-Entropy and Soft Dice loss to address class imbalance and improve overlap quality for small tumor regions, especially Enhancing Tumor (ET). The implementation supports both raw BraTS folder structure and nnU-Net `dataset.json` format, includes smart foreground cropping and basic augmentation, and is trained using PyTorch Lightning for reproducibility.

Experimental results from the project logs show a best validation average Dice score of **0.8275** at epoch **82** (`best_model_epoch=82_dice=0.8275.ckpt`), with per-region Dice of **ET 0.7785**, **TC 0.8114**, and **WT 0.8925**. The system also includes sliding-window inference and qualitative visual reports for axial/coronal/sagittal views with prediction-versus-ground-truth overlays. Overall, the project demonstrates a robust and extensible baseline for 3D brain tumor segmentation, while highlighting clear opportunities for future improvement in ET boundary precision and cross-dataset generalization.

## Acknowledgements
I would like to thank my supervisor, module instructors, and peers for guidance and feedback throughout this project. I also acknowledge the BraTS challenge organizers for providing benchmark datasets and the open-source community for PyTorch, PyTorch Lightning, and related scientific imaging tools used in this implementation.

## Table of Contents
1. Introduction  
2. Literature Review  
3. Design  
4. Implementation and Testing  
5. Project Evaluation  
6. Further Work and Conclusion  
7. Glossary  
8. Table of Abbreviations  
9. References / Bibliography  
10. Appendix A

## Table of Figures
- Figure 1. Training metrics across epochs (`my_lsnet_3d_results.png`)  
- Figure 2. Example inference report with multi-view overlays (`test_outputs/BraTS_Test_471_Final.png`)  

## Introduction
Brain tumor segmentation from MRI is a critical task in computer-aided diagnosis and treatment planning. Manual annotation is time-consuming and subject to inter-observer variation, motivating the need for automatic and reliable segmentation systems. The BraTS benchmark defines three clinically important regions: Enhancing Tumor (ET), Tumor Core (TC), and Whole Tumor (WT).

The aim of this project is to build a practical 3D segmentation pipeline tailored for BraTS data, with the following objectives:
- Design an end-to-end model that ingests four MRI modalities (T1, T1ce, T2, FLAIR) and outputs voxel-wise tumor labels.
- Improve learning on small and imbalanced tumor regions through weighted losses and tumor-focused sampling.
- Provide a reproducible training and testing workflow with quantitative metrics and qualitative visualization.

The final system is implemented in Python using PyTorch and PyTorch Lightning, with support for both training and full-volume inference.

## Literature Review
Recent progress in medical image segmentation has been driven by encoder-decoder architectures and transformer/hybrid designs. U-Net-style skip connections remain widely used because they preserve fine spatial detail while capturing semantic context. For BraTS tasks, volumetric (3D) models generally outperform 2D models by exploiting inter-slice continuity.

Key concepts relevant to this project include:
- **Dice-based optimization:** Dice overlap is strongly correlated with segmentation quality on imbalanced medical data.
- **Class imbalance handling:** Weighted cross-entropy and region-aware sampling improve rare class learning.
- **Attention mechanisms:** Channel and spatial attention (e.g., CBAM) can enhance feature selectivity for lesion regions.
- **Multi-scale supervision:** Auxiliary outputs at different resolutions improve gradient flow and convergence stability.

This project aligns with those directions by combining an LSNet3D-style backbone with CBAM-based decoder attention and deep supervision.

## Design
### 3.1 System Overview
The pipeline consists of:
- Data loading and preprocessing for BraTS volumes.
- LSNet3D encoder for hierarchical 3D feature extraction.
- CBAM-enhanced decoder with skip connections and auxiliary outputs.
- Combined loss (weighted CE + Soft Dice).
- BraTS metric evaluation (Dice for ET/TC/WT; optional HD95).
- Sliding-window full-volume inference and report visualization.

### 3.2 Data Design
Input format supports:
- Raw BraTS case folders (`*_t1.nii`, `*_t1ce.nii`, `*_t2.nii`, `*_flair.nii`, `*_seg.nii`).
- nnU-Net format (`imagesTr`, `labelsTr`, `dataset.json`).

Preprocessing includes:
- Label remap to 4 classes: background=0, NCR/NET=1, ED=2, ET=3 (mapping original label 4 to 3).
- Per-modality z-score normalization on non-zero brain voxels.
- Tumor-aware cropping (`64x64x64`) with 70% probability centered near foreground.
- Random flip and light Gaussian noise augmentation during training.

### 3.3 Model Design
- **Encoder:** `LSNet3D` with dynamic large/small kernel components (`LKP3D`, `SKA3D`), residual FFN blocks, and attention at deeper stage.
- **Decoder:** Progressive 3D upsampling with skip fusion and CBAM at each stage.
- **Deep supervision:** Auxiliary logits at 8³, 16³, and 32³ scales, plus main 64³ output.

### 3.4 Loss and Optimization Design
- Main criterion: `DS_UNETR_PlusPlus_Loss = CrossEntropy + SoftDice`.
- Class weights: `[0.1, 1.0, 1.0, 5.0]` to reduce background dominance and emphasize ET.
- Total training loss: `L_main + 0.3 * mean(L_aux)`.
- Optimizer: AdamW (`lr=1e-4`, `weight_decay=1e-5`).
- Scheduler: ReduceLROnPlateau (monitor `val_dice_avg`).

## Implementation and Testing
### 4.1 Development Environment
- Language: Python
- Frameworks: PyTorch, PyTorch Lightning
- Medical I/O: nibabel
- Plotting: matplotlib
- Data split: scikit-learn `train_test_split`

### 4.2 Training Configuration
- Batch size: 4
- Patch/crop size: 64³
- Max epochs: 200
- Precision: mixed-precision (16-bit) on GPU, 32-bit on CPU
- Validation checkpoint: best by `val_dice_avg`

### 4.3 Testing Workflow
Testing is performed using `test_model.py`:
- Loads Lightning checkpoint weights.
- Runs sliding-window inference (`window_size=128`, `overlap=0.5`) on full volumes.
- Converts predictions to BraTS regions and computes ET/TC/WT Dice.
- Optionally computes HD95 if SciPy is installed.
- Saves full visual reports in `test_outputs/`.

### 4.4 Verification and Artifacts
- Training metrics plot: `my_lsnet_3d_results.png`
- Example qualitative output: `test_outputs/BraTS_Test_471_Final.png`
- Saved best checkpoint: `weight/BraTS/best_model_epoch=82_dice=0.8275.ckpt`

## Project Evaluation
### 5.1 Quantitative Results (Validation)
From `lightning_logs/version_7/metrics.csv` (epoch-averaged):
- Best epoch: **82**
- `val_dice_avg`: **0.8275**
- `val_dice_tc`: **0.8114**
- `val_dice_wt`: **0.8925**
- `val_dice_et`: **0.7785**
- `val_loss`: **0.9264**

Final logged epoch (111) indicates slight regression versus best checkpoint:
- `val_dice_avg`: 0.8106
- `val_dice_tc`: 0.8012
- `val_dice_wt`: 0.8764
- `val_dice_et`: 0.7543

This supports selecting the best validation checkpoint rather than the final epoch for deployment.

### 5.2 Qualitative Results
Visual inspection of generated reports shows:
- WT region is generally captured well and consistently.
- TC is segmented with good spatial coherence.
- ET remains the most challenging region due to small volume and fuzzy boundaries.

### 5.3 Strengths
- End-to-end 3D pipeline with reproducible train/val/test setup.
- Robust support for multiple BraTS data layouts.
- Practical enhancements for imbalance: weighted loss + foreground-aware cropping + deep supervision.
- Good WT/TC performance and interpretable outputs.

### 5.4 Limitations
- ET sensitivity and boundary precision can still vary across cases.
- Current augmentation strategy is basic.
- No cross-validation or external dataset testing yet.
- HD95 evaluation depends on optional SciPy setup and may be expensive.

## Further Work and Conclusion
### 6.1 Further Work
- Introduce stronger augmentation (elastic deformation, intensity shift, gamma/contrast transforms).
- Add post-processing (connected-component filtering, region consistency rules).
- Perform k-fold cross-validation for more reliable generalization estimates.
- Explore larger crop sizes or memory-efficient full-resolution training.
- Benchmark against nnU-Net and transformer-based baselines under identical splits.

### 6.2 Conclusion
This project successfully implemented and evaluated a 3D LSNet-based brain tumor segmentation system for BraTS data. The proposed combination of LSNet3D encoder, CBAM decoder, and deep supervision achieved a best validation average Dice of 0.8275, with strong WT and TC performance and acceptable ET results. The codebase is modular, reproducible, and suitable for future extension toward higher clinical robustness and stronger comparative benchmarking.

## Glossary
- **Segmentation:** Pixel/voxel-wise classification of structures in an image.
- **Voxel:** A volumetric pixel in 3D space.
- **Logits:** Raw model outputs before softmax/sigmoid normalization.
- **Deep Supervision:** Auxiliary losses attached to intermediate layers.
- **Sliding Window Inference:** Patch-wise prediction with overlap for full-size volumes.
- **Foreground Cropping:** Sampling patches centered around non-background regions.

## Table of Abbreviations
- **BraTS:** Brain Tumor Segmentation Challenge
- **MRI:** Magnetic Resonance Imaging
- **ET:** Enhancing Tumor
- **TC:** Tumor Core
- **WT:** Whole Tumor
- **CBAM:** Convolutional Block Attention Module
- **CE:** Cross-Entropy
- **DSC/Dice:** Dice Similarity Coefficient
- **HD95:** 95th percentile Hausdorff Distance
- **NCR/NET:** Necrotic and Non-Enhancing Tumor Core
- **ED:** Peritumoral Edema

## References / Bibliography
1. Menze, B. H., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993-2024.  
2. Bakas, S., et al. (2018). Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge. *arXiv preprint arXiv:1811.02629*.  
3. Isensee, F., et al. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18, 203-211.  
4. Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. In *ECCV 2018*.  
5. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). In *ICLR 2019*.  
6. Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between Species. *Ecology*, 26(3), 297-302.  
7. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *NeurIPS 2019*.  
8. Falcon, W., & The PyTorch Lightning Team. (2019). PyTorch Lightning. https://www.pytorchlightning.ai/

## Appendix A: First Appendix
### A.1 Main Commands
```bash
# Train
python config/train.py

# Evaluate + generate qualitative reports
python test_model.py

# Plot training curves
python plot_metrics.py
```

### A.2 Repository Paths Used in This Report
- `config/train.py`
- `config/segmentor.py`
- `datasets/datasets.py`
- `module/model/lsnet3d.py`
- `module/model/decoder3d.py`
- `loss/loss.py`
- `metrics/metrics.py`
- `test_model.py`
- `lightning_logs/version_7/metrics.csv`

"""
Generator for Deep_Learning_Diabetes_Classifier_Notebook_Documentation.docx
Exhaustive, Publication-Grade Documentation of the Jupyter Notebook (Deep_Learning_Diabetes_Classifier.ipynb)
"""

import os
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOC_PATH = Path("Deep_Learning_Diabetes_Classifier_Notebook_Documentation.docx")
NB_IMG_DIR = Path("notebook_images")
FIG_DIR = Path("figures")

doc = Document()

# Page Setup: Standard Letter, 1-inch margins
for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.different_first_page_header_footer = True
    
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run("Deep Learning Diabetes Classifier — Notebook Master Documentation")
    hrun.font.name = "Arial"
    hrun.font.size = Pt(8.5)
    hrun.font.color.rgb = RGBColor(120, 144, 156)
    
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = fp.add_run("Deep_Learning_Diabetes_Classifier.ipynb | Experimental & Research Master Report")
    frun.font.name = "Arial"
    frun.font.size = Pt(8.5)
    frun.font.color.rgb = RGBColor(120, 144, 156)

COLOR_PRIMARY = RGBColor(27, 54, 93)     # Deep Navy
COLOR_SECONDARY = RGBColor(43, 84, 126)  # Slate Blue
COLOR_ACCENT = RGBColor(153, 0, 0)       # Crimson Accent
COLOR_TEXT = RGBColor(34, 34, 34)        # Dark Charcoal
COLOR_MUTED = RGBColor(100, 116, 139)    # Muted Slate

HEX_PRIMARY = "1B365D"
HEX_LIGHT_BG = "F4F7FA"
HEX_BORDER = "CBD5E1"
HEX_ACCENT = "990000"
HEX_CALLOUT_BG = "EFF6FF"
HEX_CALLOUT_BORDER = "2563EB"

normal_style = doc.styles['Normal']
normal_style.font.name = 'Arial'
normal_style.font.size = Pt(10.5)
normal_style.font.color.rgb = COLOR_TEXT
normal_style.paragraph_format.line_spacing = 1.15
normal_style.paragraph_format.space_after = Pt(6)

def set_cell_background(cell, hex_color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl_pr.append(borders)

def add_h1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(17)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return p

def add_h2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY
    return p

def add_h3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Arial"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(62, 92, 118)
    return p

def add_callout(title, text, callout_type="info"):
    border_color = HEX_CALLOUT_BORDER if callout_type == "info" else HEX_ACCENT
    bg_color = HEX_CALLOUT_BG if callout_type == "info" else "FEF2F2"
    
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tc_pr.append(borders)
    
    cp = cell.paragraphs[0]
    cp.paragraph_format.space_after = Pt(2)
    run_title = cp.add_run(f"📌 {title}\n" if callout_type == "info" else f"⚠️ {title}\n")
    run_title.font.bold = True
    run_title.font.size = Pt(10)
    run_title.font.color.rgb = RGBColor(37, 99, 235) if callout_type == "info" else RGBColor(185, 28, 28)
    
    run_text = cp.add_run(text)
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = COLOR_TEXT
    
    doc.add_paragraph()

def add_image(image_path, caption_text, width_inches=6.2):
    p = Path(image_path)
    if p.exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        run = p_img.add_run()
        run.add_picture(str(p), width=Inches(width_inches))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(12)
        run_cap = p_cap.add_run(caption_text)
        run_cap.font.size = Pt(9.0)
        run_cap.font.italic = True
        run_cap.font.color.rgb = COLOR_MUTED

def create_styled_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for r in p.runs:
            r.font.bold = True
            r.font.size = Pt(9.5)
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_col = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=90, bottom=90, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                r.font.size = Pt(9.0)
                r.font.color.rgb = COLOR_TEXT
                
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)
                
    doc.add_paragraph()
    return table

print("Building Title Page...")
# ===========================================================================
# COVER PAGE / TITLE PAGE
# ===========================================================================
title_p = doc.add_paragraph()
title_p.paragraph_format.space_before = Pt(80)
title_p.paragraph_format.space_after = Pt(8)
run_title = title_p.add_run("DEEP LEARNING DIABETES CLASSIFIER")
run_title.font.name = "Arial"
run_title.font.size = Pt(28)
run_title.font.bold = True
run_title.font.color.rgb = COLOR_PRIMARY

sub_p = doc.add_paragraph()
sub_p.paragraph_format.space_after = Pt(24)
run_sub = sub_p.add_run(
    "Exhaustive Research & Engineering Documentation of the Jupyter Notebook:\n"
    "Deep_Learning_Diabetes_Classifier.ipynb\n"
    "From Raw Cohorts to ResNet Architecture, MixUp Training, Benchmarks, ROC Curves & Permutation Sensitivity"
)
run_sub.font.size = Pt(13.0)
run_sub.font.italic = True
run_sub.font.color.rgb = COLOR_SECONDARY

bar_table = doc.add_table(rows=1, cols=1)
bar_cell = bar_table.cell(0, 0)
bar_cell.width = Inches(6.5)
set_cell_background(bar_cell, HEX_PRIMARY)
set_cell_margins(bar_cell, top=20, bottom=20, left=20, right=20)
doc.add_paragraph()

meta_p = doc.add_paragraph()
meta_p.paragraph_format.space_before = Pt(30)
meta_p.paragraph_format.space_after = Pt(4)
meta_p.add_run("NOTEBOOK ARTIFACT SPECIFICATIONS & SUMMARY\n").font.bold = True

meta_table_data = [
    ["Notebook File", "Deep_Learning_Diabetes_Classifier.ipynb (35 Executed Cells)"],
    ["Dataset Examined", "cleaned_diabetes_data.csv (1,898 Patient Cohorts)"],
    ["Model Architecture", "DiabetesTabularResNet (Dense Residual Skip Connections, GELU, BatchNorm1d)"],
    ["Evaluation Accuracy", "98.60% Out-of-Sample Test Accuracy (Macro ROC-AUC: 0.9994 | Macro F1: 0.9843)"],
    ["Data Partitions", "3-Way Stratified: 70% Train (1,328), 15% Validation (285), 15% Test (285)"],
    ["Regularization Pipeline", "MixUp Tabular Augmentation (α=0.20), Label Smoothing (ε=0.03), Dropout (0.20/0.15)"],
    ["Optimization Suite", "AdamW (LR: 3e-3, Weight Decay: 1e-4) + CosineAnnealingLR (100 Epochs)"],
    ["Benchmark Models", "Random Forest, Extra Trees, RBF SVM, Logistic Regression, Deep ResNet"],
    ["Interpretability", "Permutation Feature Importance Sensitivity Analysis across all 13 Features"],
    ["Framework Parity", "PyTorch 2.x Primary Implementation + Bonus TensorFlow/Keras Equivalent Code"]
]
create_styled_table(["Artifact Parameter", "Notebook Execution Detail"], meta_table_data, [2.3, 4.2])

doc.add_page_break()

# ===========================================================================
# TABLE OF CONTENTS
# ===========================================================================
add_h1("Table of Contents — Notebook Experimental Pipeline")

toc_items = [
    ("Step 1", "Environment Initialization, Seeds & Hardware Acceleration Setup"),
    ("Step 2", "Global Configuration Schema (Config Class & Hyperparameters)"),
    ("Step 3", "Dataset Ingestion, Verification & Baseline Clinical Profiling"),
    ("Step 4", "Clinical Biomarker Engineering & Interaction Feature Derivation"),
    ("Step 5", "3-Way Stratified Partitioning (Train 70%, Val 15%, Test 15%)"),
    ("Step 6", "Feature Normalization Pipeline (StandardScaler Leakage Prevention)"),
    ("Step 7", "DiabetesTabularResNet Deep Neural Architecture Implementation"),
    ("Step 8", "Optimization Setup: Balanced Weights, Label Smoothing & AdamW"),
    ("Step 9", "Training & Validation Epoch Loop with MixUp & State Recovery"),
    ("Step 10", "Training Convergence, Loss Trajectories & Diagnostic Curves (Graph Results)"),
    ("Step 11", "Out-of-Sample Test Evaluation, Metrics & Confusion Matrix (Results)"),
    ("Step 12", "Multi-Class ROC Curves & Precision-Recall Trajectories (AUC = 0.9994)"),
    ("Step 13", "5-Fold Cross-Validation & Multi-Model Tabular Benchmark Suite"),
    ("Step 14", "Permutation Feature Importance & Clinical Sensitivity Analysis"),
    ("Step 15", "Exporting Production Model Artifacts, Scaler & JSON Metadata"),
    ("Step 16", "Real-Time Patient Clinical Inference Engine & Profile Case Studies"),
    ("Bonus Step 17", "Equivalent TensorFlow / Keras Multi-Framework Implementation")
]

toc_table = doc.add_table(rows=len(toc_items), cols=2)
toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for idx, (ch, name) in enumerate(toc_items):
    r_cells = toc_table.rows[idx].cells
    r_cells[0].width = Inches(1.5)
    r_cells[1].width = Inches(5.0)
    set_cell_margins(r_cells[0], top=50, bottom=50, left=80, right=80)
    set_cell_margins(r_cells[1], top=50, bottom=50, left=80, right=80)
    p0 = r_cells[0].paragraphs[0]
    p0.add_run(ch).font.bold = True
    p0.runs[0].font.color.rgb = COLOR_PRIMARY
    p1 = r_cells[1].paragraphs[0]
    p1.add_run(name)

doc.add_page_break()

# ===========================================================================
# STEP 1 & 2: SETUP & CONFIGURATION
# ===========================================================================
add_h1("Step 1 & Step 2: System Setup, Environment & Configuration Schema")

doc.add_paragraph(
    "Cells 0 to 4 in Deep_Learning_Diabetes_Classifier.ipynb establish the global execution environment. "
    "To ensure deterministic reproducibility across random data splits, neural weight initialization, and MixUp augmentation, "
    "a rigorous seed_everything(42) routine is executed across Python's hash seed, NumPy, PyTorch CPU, and PyTorch CUDA backends."
)

doc.add_paragraph(
    "All experiment hyperparameters, filepaths, and clinical mappings are codified in a centralized Config object:"
)

cfg_table = [
    ["DATA_PATH", "cleaned_diabetes_data.csv", "Path to the 1,898-row clinical registry"],
    ["TARGET_COL", "diabetic", "Target outcome: Low (0), Medium (1), High (2)"],
    ["HIDDEN_DIM", "128", "Dimensionality of latent manifold and residual blocks"],
    ["BATCH_SIZE", "32", "Mini-batch size for DataLoader pipelines"],
    ["EPOCHS", "100", "Maximum training epochs with cosine learning rate decay"],
    ["LEARNING_RATE", "3e-3 (0.003)", "Initial peak learning rate for AdamW optimizer"],
    ["WEIGHT_DECAY", "1e-4 (0.0001)", "Decoupled L2 regularization parameter"],
    ["LABEL_SMOOTHING", "0.03 (3%)", "Soft target probability redistribution parameter"],
    ["DROPOUT_RATE", "0.20 / 0.15 / 0.10", "Asymmetric dropout regularization across layers"],
    ["MIXUP_ALPHA", "0.20", "Beta distribution shape parameter for MixUp augmentation"],
    ["MIXUP_PROB", "0.50 (50%)", "Probability of applying MixUp per training mini-batch"],
    ["DATA_SPLIT", "70% / 15% / 15%", "Stratified Train (1,328) / Val (285) / Test (285)"]
]
create_styled_table(["Config Parameter", "Configured Value", "Operational Rationale"], cfg_table, [2.0, 1.8, 2.7])

# ===========================================================================
# STEP 3: DATASET INGESTION & PROFILING
# ===========================================================================
add_h1("Step 3: Dataset Ingestion, Verification & Baseline Clinical Profiling")

doc.add_paragraph(
    "In Cells 5 and 6, cleaned_diabetes_data.csv is loaded into Pandas. The dataset contains 1,898 adult patient records across "
    "9 raw clinical metrics and 1 multi-class ground truth target ('diabetic')."
)

eda_table_data = [
    ["Age", "int64", "46.73 ± 13.67", "21 – 80 years", "Chronological aging risk biomarker"],
    ["gender", "int64", "0.38 ± 0.48", "0 (Female) / 1 (Male)", "Biological sex distribution (38% Male, 62% Female)"],
    ["Glucose", "float64", "137.28 ± 55.88", "7.74 – 602.28 mg/dL", "Fasting blood glucose enzymatic assay"],
    ["weight", "float64", "54.06 ± 10.07", "20.0 – 95.0 kg", "Body mass assessed via calibrated medical scales"],
    ["height", "float64", "1.56 ± 0.08", "1.30 – 1.92 m", "Standing stature measured via stadiometer"],
    ["bmi", "float64", "22.31 ± 4.06", "10.02 – 45.86 kg/m²", "Calculated Body Mass Index (weight / height²)"],
    ["systolic_bp", "int64", "134.53 ± 22.16", "86 – 226 mmHg", "Arterial pressure during ventricular contraction"],
    ["diastolic_bp", "int64", "82.07 ± 12.54", "51 – 119 mmHg", "Arterial pressure during ventricular relaxation"],
    ["Hba1c", "float64", "6.05 ± 1.14", "4.50 – 9.00 %", "Gold-standard 3-month glycated hemoglobin assay"],
    ["diabetic", "object", "Low / Med / High", "3 Discrete Tiers", "Ground truth ADA clinical risk stratification"]
]
create_styled_table(["Feature Column", "Data Type", "Mean ± Std", "Range / Values", "Clinical Diagnostic Role"], eda_table_data, [1.4, 0.9, 1.4, 1.4, 1.4])

# ===========================================================================
# STEP 4: CLINICAL FEATURE ENGINEERING
# ===========================================================================
add_h1("Step 4: Clinical Biomarker Engineering & Interaction Derivation")

doc.add_paragraph(
    "In Cell 8 of the notebook, four physiological interaction features are derived to enrich the raw 9-dimensional patient vector "
    "into a 13-dimensional continuous manifold:"
)

doc.add_paragraph(
    "1. Mean Arterial Pressure (MAP):\n"
    "   MAP = diastolic_bp + (systolic_bp - diastolic_bp) / 3.0\n"
    "   • Physiological Basis: Measures the perfusion pressure driving blood through capillary beds during a full cardiac cycle (1/3 systole, 2/3 diastole).\n\n"
    "2. Pulse Pressure (PP):\n"
    "   Pulse_Pressure = systolic_bp - diastolic_bp\n"
    "   • Physiological Basis: Quantifies aortic compliance. Values > 60 mmHg reflect vascular stiffening and accelerated vascular aging.\n\n"
    "3. Glucose-to-HbA1c Ratio:\n"
    "   Glucose_HbA1c_Ratio = Glucose / (Hba1c + 1e-5)\n"
    "   • Physiological Basis: Acute-to-chronic metabolic ratio. Identifies unstable glycemic spikes vs steady-state diabetes.\n\n"
    "4. Metabolic Risk Index:\n"
    "   Metabolic_Index = (bmi * Age) / 100.0\n"
    "   • Physiological Basis: Models cumulative duration of adiposity-driven peripheral insulin resistance."
)

# ===========================================================================
# STEP 5 & 6: DATA SPLITTING & SCALING
# ===========================================================================
add_h1("Step 5 & Step 6: 3-Way Stratified Partitioning & Leakage-Free Scaling")

doc.add_paragraph(
    "In Cells 10 and 12, the notebook implements a strict two-stage stratified data partition to guarantee that zero validation or test parameters leak into the training process:"
)

split_table_data = [
    ["1. Training Set", "1,328 patients (70.0%)", "Backpropagation, gradient descent & MixUp augmentation", "StandardScaler.fit_transform()"],
    ["2. Validation Set", "285 patients (15.0%)", "Epoch-by-epoch macro-F1 evaluation & checkpoint restoration", "StandardScaler.transform()"],
    ["3. Test Set", "285 patients (15.0%)", "Final untouched holdout evaluation (98.60% accuracy achieved)", "StandardScaler.transform()"]
]
create_styled_table(["Partition", "Cohort Size", "Role in Pipeline", "Scaling Isolation Rule"], split_table_data, [1.5, 1.8, 1.8, 1.4])

# ===========================================================================
# STEP 7: RESNET NEURAL ARCHITECTURE
# ===========================================================================
add_h1("Step 7: Deep Tabular ResNet (DiabetesTabularResNet) Neural Architecture")

doc.add_paragraph(
    "In Cell 14, the notebook defines DiabetesTabularResNet using PyTorch. The model incorporates dense residual skip connections "
    "that preserve continuous feature gradients across depth:"
)

add_image("figures/fig3_neural_architecture.png", "Figure 7.1: Deep Tabular ResNet (DiabetesTabularResNet) Block Architecture Schematic.")

doc.add_paragraph(
    "• Input Projection: Maps the 13 normalized features to 128 dimensions using Linear(13, 128) + BatchNorm1d + GELU.\n"
    "• Tabular Residual Block 1: Two Linear(128, 128) transformations with BatchNorm, GELU, and Dropout (0.20 / 0.15), bypassed by an identity skip connection: y = x + F(x).\n"
    "• Tabular Residual Block 2: Second residual refinement block with identical skip addition.\n"
    "• Classifier Head: Linear(128, 64) + BatchNorm1d + GELU + Dropout(0.10) -> Linear(64, 3).\n"
    "• Total Parameters: 76,867 trainable parameters initialized via Kaiming Normal (He) initialization."
)

# ===========================================================================
# STEP 8 & 9: OPTIMIZATION & TRAINING LOOP
# ===========================================================================
add_h1("Step 8 & Step 9: Optimization Setup & Epoch Training with MixUp")

doc.add_paragraph(
    "In Cells 16 and 18, the optimization pipeline is configured and executed:"
)

doc.add_paragraph(
    "1. Balanced Class Weights:\n"
    "   W_Low = 0.7744, W_Medium = 1.0406, W_High = 1.3376 applied to CrossEntropyLoss.\n\n"
    "2. Label Smoothing (ε = 0.03):\n"
    "   Replaces hard one-hot targets with soft distributions, preventing overconfident misclassifications on borderline HbA1c values.\n\n"
    "3. MixUp Tabular Data Augmentation:\n"
    "   Applied with 50% probability per mini-batch: x_mix = λ x_i + (1 - λ) x_j, where λ ~ Beta(0.2, 0.2).\n\n"
    "4. AdamW & Cosine Annealing Learning Rate:\n"
    "   Optimizer: AdamW(lr=3e-3, weight_decay=1e-4) paired with CosineAnnealingLR(T_max=100, eta_min=1e-5).\n\n"
    "5. Dynamic Checkpoint Recovery:\n"
    "   At each epoch, the validation macro-F1 score is evaluated. The state_dict achieving peak validation performance is saved and restored."
)

# ===========================================================================
# STEP 10: TRAINING CONVERGENCE DIAGNOSTICS (GRAPH RESULT)
# ===========================================================================
add_h1("Step 10: Training Convergence & Loss Diagnostics (Graph Results)")

doc.add_paragraph(
    "Cell 20 visualizes the training convergence diagnostics across the 100 training epochs, plotting Training Loss vs Validation Loss, "
    "Training Accuracy vs Validation Accuracy, and the Cosine Annealing learning rate decay:"
)

add_image("notebook_images/nb_cell_20_img_1.png", "Figure 10.1: Notebook Training Convergence Diagnostics — Loss Trajectory, Accuracy Curve & Learning Rate Decay over 100 Epochs.")

doc.add_paragraph(
    "Diagnostic Observations:\n"
    "• Loss Convergence: Training loss steadily declines from 0.7437 at Epoch 1 to < 0.28, while validation loss stabilizes around 0.21, exhibiting zero divergence or overfitting.\n"
    "• Accuracy Trajectory: Validation accuracy rapidly surges from 91.58% at Epoch 1 to > 98.5% within 30 epochs.\n"
    "• Learning Rate Annealing: Cosine schedule smoothly scales the learning rate from 3.00e-3 down to 1.00e-5, allowing fine-grained convergence in the final 20 epochs."
)

# ===========================================================================
# STEP 11: TEST EVALUATION & CONFUSION MATRIX (GRAPH RESULT)
# ===========================================================================
add_h1("Step 11: Out-of-Sample Test Evaluation & Confusion Matrix (Graph Results)")

doc.add_paragraph(
    "In Cell 22, the restored optimal checkpoint is evaluated on the untouched 285-patient test set, achieving 98.60% accuracy and 0.9843 Macro F1:"
)

test_rep_data = [
    ["Low Risk (< 5.7%)", "1.0000 (100.0%)", "0.9919 (99.2%)", "0.9959 (99.6%)", "123 patients"],
    ["Medium Risk (5.7% – 6.4%)", "0.9579 (95.8%)", "1.0000 (100.0%)", "0.9785 (97.9%)", "91 patients"],
    ["High Risk (≥ 6.5%)", "1.0000 (100.0%)", "0.9577 (95.8%)", "0.9784 (97.8%)", "71 patients"],
    ["Macro Average", "0.9859 (98.6%)", "0.9832 (98.3%)", "0.9843 (98.4%)", "285 patients"],
    ["Weighted Average", "0.9866 (98.7%)", "0.9860 (98.6%)", "0.9860 (98.6%)", "285 patients"]
]
create_styled_table(["Class", "Precision", "Recall", "F1-Score", "Test Support"], test_rep_data, [2.0, 1.1, 1.1, 1.1, 1.2])

add_image("notebook_images/nb_cell_22_img_1.png", "Figure 11.1: Notebook Confusion Matrix Heatmap on Untouched Test Set (N = 285).")

add_callout(
    "Clinical Safety Assurance",
    "Zero high-risk diabetic patients were misclassified as Low Risk (0/71 misses). The 3 borderline cases were classified as Medium Risk (Pre-diabetic), "
    "ensuring that every diabetic patient receives clinical follow-up.",
    "warning"
)

# ===========================================================================
# STEP 12: MULTI-CLASS ROC & PR CURVES (GRAPH RESULT)
# ===========================================================================
add_h1("Step 12: Multi-Class ROC & Precision-Recall Curves (Graph Results)")

doc.add_paragraph(
    "Cell 24 computes the One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) and Precision-Recall curves across all three classes, "
    "yielding an overall Macro-Averaged ROC-AUC of 0.9994:"
)

add_image("notebook_images/nb_cell_24_img_1.png", "Figure 12.1: Multi-Class One-vs-Rest ROC & Precision-Recall Curves (Macro ROC-AUC = 0.9994).")

# ===========================================================================
# STEP 13: MULTI-MODEL BENCHMARK SUITE (COMPARISON)
# ===========================================================================
add_h1("Step 13: 5-Fold Cross-Validation & Multi-Model Benchmark Suite (Comparison)")

doc.add_paragraph(
    "In Cell 26, the notebook benchmarks Deep Tabular ResNet against 4 standard tabular machine learning algorithms using 5-Fold Stratified Cross-Validation on the training data and holdout validation:"
)

nb_benchmark_data = [
    ["Random Forest (300 Trees)", "1.0000 ± 0.0000", "100.00%", "1.0000", "1.0000", "1.0000"],
    ["Deep Tabular ResNet (Ours)", "0.9895 ± 0.0040", "98.95%", "0.9895", "0.9895", "0.9994"],
    ["Extra Trees (300 Trees)", "0.9856 ± 0.0061", "98.95%", "0.9900", "0.9895", "0.9997"],
    ["Logistic Regression (L2)", "0.9818 ± 0.0038", "99.65%", "0.9964", "0.9968", "1.0000"],
    ["RBF SVM (C=100)", "0.9553 ± 0.0163", "95.44%", "0.9581", "0.9549", "0.9968"]
]
create_styled_table(["Model Architecture", "5-Fold CV Macro F1", "Val Accuracy", "Val Precision", "Val Macro F1", "Val Macro ROC-AUC"], nb_benchmark_data, [2.1, 1.4, 0.9, 0.9, 0.9, 1.0])

add_image("notebook_images/nb_cell_26_img_1.png", "Figure 13.1: Multi-Model Benchmark Comparison — 5-Fold Cross-Validation Macro-F1 across 5 Machine Learning Architectures.")

# ===========================================================================
# STEP 14: PERMUTATION FEATURE IMPORTANCE (GRAPH RESULT)
# ===========================================================================
add_h1("Step 14: Permutation Feature Importance & Sensitivity Diagnostics (Graph Results)")

doc.add_paragraph(
    "In Cell 28, the notebook performs Permutation Feature Importance on the test set. Each feature column is shuffled independently to break its relationship with the target, and the resulting drop in test accuracy is measured:"
)

add_image("notebook_images/nb_cell_28_img_1.png", "Figure 14.1: Permutation Feature Importance Bar Chart — Accuracy Drop when Individual Features are Permuted on Test Cohorts.")

doc.add_paragraph(
    "Clinical Sensitivity Findings:\n"
    "• Primary Biomarkers: Glycated Hemoglobin (HbA1c) and Fasting Glucose are the dominant predictive drivers. Shuffling HbA1c induces the largest drop in classification accuracy, confirming adherence to ADA clinical standards.\n"
    "• Engineered Synergy: The derived Glucose-to-HbA1c Ratio, Mean Arterial Pressure (MAP), and Metabolic Risk Index provide critical complementary signal for borderline pre-diabetic patients."
)

# ===========================================================================
# STEP 15 & 16: ARTIFACT EXPORT & INFERENCE ENGINE
# ===========================================================================
add_h1("Step 15 & Step 16: Production Artifact Export & Real-Time Patient Inference")

doc.add_paragraph(
    "In Cell 30, the notebook exports the production artifacts required by our FastAPI service:\n"
    "1. diabetes_deep_model.pt: Serialized PyTorch state dictionary (331 KB).\n"
    "2. deep_model_scaler.pkl: Fitted StandardScaler object.\n"
    "3. deep_model_metadata.json: Metadata configuration schema.\n"
    "4. diabetes_deep_model_bundle.pkl: Combined model and scaler bundle."
)

doc.add_paragraph(
    "In Cell 32, a self-contained predict_patient_risk() inference function is implemented and verified on three patient clinical archetypes:"
)

patient_case_data = [
    ["Low Risk Profile", "Age 32, Male, Glucose 88, HbA1c 4.9%, BMI 22.2", "Low Risk (92.29% Confidence)", "Normal thresholds. Annual wellness check."],
    ["Medium Risk Profile", "Age 48, Female, Glucose 118, HbA1c 6.0%, BMI 29.0", "Medium Risk (96.47% Confidence)", "Pre-diabetic. Lifestyle intervention indicated."],
    ["High Risk Profile", "Age 56, Male, Glucose 175, HbA1c 8.4%, BMI 30.4", "High Risk (98.81% Confidence)", "Diabetic. Physician evaluation & pharmacological plan."]
]
create_styled_table(["Patient Archetype", "Clinical Indicators Input", "ResNet Output", "Clinical Recommendation"], patient_case_data, [1.5, 2.0, 1.5, 1.5])

# Save Document
doc.save(str(DOC_PATH))
print(f"Master Notebook Word Document successfully created and saved to: {DOC_PATH.resolve()}")

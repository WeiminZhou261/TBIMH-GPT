# TBIMH-GPT: Few-Shot Learning Prognostic Diagnosis of Type B Intramural Hematoma via Multimodal Large Language Model

> **Notice:** This repository contains the **training code** for TBIMH-GPT, covering **Stage 1 (Medical Domain Alignment)** and **Stage 2 (TBIMH-specific RLHF fine-tuning)**.  
> Inference models and evaluation tools will be **released after the paper is accepted**. The project is currently under peer review.

---

## 📘 Overview

**TBIMH-GPT** is a multimodal large language model framework designed for **prognostic diagnosis of Type B Intramural Hematoma (TBIMH)**.  
It introduces a **two-stage fine-tuning strategy** to adapt a general MLLM (e.g., LLaVA) to the TBIMH prognostic diagnosis task.

### 🔍 Highlights

- **Stage 1: Medical Domain Alignment**
  - Biomedical concept alignment
  - General medical instruction tuning
  - Medical QA fine-tuning
- **Stage 2: TBIMH-specific RLHF**
  - Reinforcement Learning from Human Feedback (RLHF)
  - Incorporation of real-world TBIMH clinical case feedback
  - Improved prognostic accuracy under limited data conditions

---

## ⚙️ Contents

```
TBIMH-GPT/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ configs/
│  ├─ stage1_alignment.yaml
│  ├─ stage1_med_instruct.yaml
│  ├─ stage1_med_qa.yaml
│  ├─ stage2_tbimh_rlhf.yaml
├─ src/
│  ├─ models/
│  ├─ trainers/
│  ├─ datasets/
│  └─ utils/
├─ scripts/
│  ├─ train_stage1.sh
│  ├─ train_stage2_rlhf.sh
│  └─ prepare_data.sh
└─ docs/
   └─ figures/
```

---

## 💡 Usage

Since this release includes only training code, you may use the following as general guidance:

```bash
# Example (Stage 1 - Medical Alignment)
bash scripts/train_stage1.sh --config configs/stage1_alignment.yaml

# Example (Stage 2 - TBIMH-specific RLHF)
bash scripts/train_stage2_rlhf.sh --config configs/stage2_tbimh_rlhf.yaml
```

> Please refer to the configuration files in `configs/` for hyperparameter definitions.

---

## 🧱 Environment Setup

```bash
conda create -n tbimh-gpt python=3.10 -y
conda activate tbimh-gpt
pip install -r requirements.txt
```

Dependencies include:
- PyTorch >= 2.2
- Transformers, PEFT, Accelerate
- Bitsandbytes (for LoRA / QLoRA training)
- SimpleITK or nibabel (for CTA data processing)

---

## 🧬 Data

This repository does **not** include any patient data.  
To train TBIMH-GPT, you need access to **de-identified** CTA imaging, textual records, and structured clinical variables.

Please ensure full compliance with **ethical and privacy regulations**.


---

## 📅 Release Plan

| Component | Status |
|------------|---------|
| Stage 1 (Medical Alignment) | ✅ Open-sourced |
| Stage 2 (TBIMH RLHF) | ✅ Open-sourced |
| Inference Model | 🔒 To be released post-publication |
| Full Evaluation Toolkit | 🔒 To be released post-publication |

---

## 🧾 Acknowledgments

We thank the collaborating cardiovascular specialists for providing guidance and clinical insights, as well as the open-source community (LLaVA, HuggingFace, PEFT, etc.) for their foundational work.

---


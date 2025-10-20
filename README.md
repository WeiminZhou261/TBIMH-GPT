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

## 💡 Usage

Since this release includes only training code, you may use the following as general guidance:

```bash
# Example (Stage 1 - Medical Alignment)
bash stage1/scripts/train_stage1_alignment.sh

bash stage1/scripts/train_stage1_med_instruct.sh

bash stage1/scripts/train_stage1_med_qa.sh

# Example (Stage 2 - TBIMH-specific RLHF)
bash stage2/scripts/train_reward_model.sh

bash stage2/scripts/initialize_policy_model.sh

bash stage2/scripts/train_rl_model.sh
```

---

## 🧱 Environment Setup

```bash
conda create -n tbimh-gpt python=3.10 -y
conda activate tbimh-gpt
pip install --upgrade pip 
pip install -e .
```

---

## 🧬 Data

You can contact our corresponding author by email. After confirming the academic purpose, we will provide a fully de-identified dataset — with all patient-identifiable information removed — within two weeks for research use.

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


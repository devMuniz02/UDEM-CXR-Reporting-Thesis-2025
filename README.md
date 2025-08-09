# Chest-X-ray-Diagnosis-Automated-Reporting-using-CNNs-and-LLMs---UDEM-PEF-Thesis-Fall-2025-
Automated X-ray diagnosis (CNN) + AI report generation (LLM). UDEM PEF thesis project. PyTorch(or Tensorflow) + transformers on radiology datasets with deployment using dockers and kubernetes and kuberflow.

# 🐳 How to Run with Docker Locally

This guide walks you through running the Chest X-ray Report Generator using Docker.

---

## 🚀 Run with Docker

Follow these steps to build and run the application using Docker:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pef-thesis.git
cd pef-thesis
```
### 2. Build the Docker Image
```bash
docker compose build
```
### 3. Run the Container
```bash
docker compose up
```
-  #### 🛑 To stop it later:
    ```bash
    docker compose down
    ```
### 4. Access the App
Once the container is running, open your browser and go to http://localhost:8000

# Project

## Chest X-Ray Datasets

<img width="620" height="277" alt="image" src="https://github.com/user-attachments/assets/c4b2ea25-1c54-4ba4-8057-40bfdbdb6a1a" />

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) - Restricted-access
- [OpenI](https://pubmed.ncbi.nlm.nih.gov/26133894/)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) 
    - Stanford CheXpert competition - [Top 1% paper](https://arxiv.org/pdf/2012.03173)
- [BraX](https://physionet.org/content/brax/1.1.0/) - Restricted-access
- [CandidPTX](https://ourarchive.otago.ac.nz/esploro/outputs/dataset/CANDID-PTX/9926556140101891) - Closed access to users outside Health New Zealand
- [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)
    - [Paper](https://arxiv.org/pdf/2405.19538v1)
- [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) -> Grounded-Reporting -> [PadChest-GR](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)
    - [Paper](https://arxiv.org/pdf/2411.05085)
- [VinDR](https://www.physionet.org/content/vindr-cxr/1.0.0/) - Restricted-access
- [MIDRC](https://www.midrc.org/) - Couldn't find the exact link to download
- JF Healthcare - Not available
- [CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus)
    - [Paper](https://arxiv.org/pdf/2405.19538v1)

### Dataset used in PEF - NIH and/or CheXpert Plus

- [LINK TO DRIVE](https://drive.google.com/drive/u/0/folders/0AConL2XD0ndHUk9PVA)

## Papers

| Paper (link) | Year | Dataset (patients / images) | Model used | AUC (per-label → macro-mean) |
|---|---:|---|---|---|
| [ChestX-ray8: Hospital-Scale Chest X-ray Database and Benchmarks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) | 2017 | ChestX-ray8 (32,717 / 108,948) | ResNet-50 (CNN) | Atelectasis 0.7069; Cardiomegaly 0.8141; Effusion 0.7362; Infiltration 0.6128; Mass 0.5609; Nodule 0.7164; Pneumonia 0.6333; Pneumothorax 0.7891 → **0.696** |
| [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-rays with Deep Learning](https://arxiv.org/pdf/1711.05225) | 2017 | ChestX-ray14 (30,805 / 112,120) | DenseNet-121 (CNN) | Atelectasis 0.8094; Cardiomegaly 0.9248; Effusion 0.8638; Infiltration 0.7345; Mass 0.8676; Nodule 0.7802; Pneumonia 0.7680; Pneumothorax 0.8887; Consolidation 0.7901; Edema 0.8878; Emphysema 0.9371; Fibrosis 0.8047; Pleural Thickening 0.8062; Hernia 0.9164 → **0.841** |
| [Learning to Diagnose from Scratch by Exploiting Dependencies Among Labels](https://arxiv.org/abs/1710.10501) | 2017 | ChestX-ray14 (30,805 / 112,120) | DenseNet + LSTM (CNN+RNN) | Atelectasis 0.772; Cardiomegaly 0.904; Effusion 0.859; Infiltration 0.695; Mass 0.792; Nodule 0.717; Pneumonia 0.713; Pneumothorax 0.841; Consolidation 0.788; Edema 0.882; Emphysema 0.829; Fibrosis 0.767; Pleural Thickening 0.765; Hernia 0.914 → **0.803** |
| [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://ojs.aaai.org/index.php/AAAI/article/view/3834) | 2019 | CheXpert (65,240 / 224,316) | DenseNet-121 ensemble (CNN) | Pleural Effusion **0.97**; Atelectasis **0.85**; Cardiomegaly ≥0.90; Consolidation ≥0.90; Edema ≥0.90 → **≈0.92** (5 labels) |
| [CheXzero: Expert-level detection of pathologies from unannotated chest X-rays via self-supervised learning](https://www.nature.com/articles/s41551-022-00936-9) | 2022 | CheXpert (65,240 / 224,316; eval on expert test) | CLIP-style image-text (ViL) | Pleural Effusion 0.958; Edema 0.961; Atelectasis 0.798; Consolidation 0.871; Cardiomegaly 0.898 → **0.889** |
| [SwinCheX: Multi-label Classification on Chest X-ray Images Using Vision Transformers](https://arxiv.org/pdf/2206.04246) | 2022 | ChestX-ray14 (~30,850 / 112,120) | Swin Transformer + multi-head MLP (ViT) | Cardiomegaly 0.875; Emphysema 0.914; Edema 0.848; Hernia 0.855; Pneumothorax 0.871; Effusion 0.824; Mass 0.822; Fibrosis 0.826; Atelectasis 0.781; Consolidation 0.748; Pleural Thickening 0.778; Nodule 0.780; Pneumonia 0.713; Infiltration 0.701 → **0.810** |
| [LT-ViT: A Vision Transformer for Multi-label Chest X-ray Classification](https://ar5iv.org/pdf/2311.07263) | 2023 | NIH-CXR14 (30,805 / 112,120); CheXpert (65,240 / 224,316) | ViT-Small + Label Tokens (ViT) | NIH-14 **0.8198**; CheXpert-5 **0.8890**; CheXpert-13 **0.7734** |
| [Anatomy-XNet: An Anatomy-Aware CNN for Thoracic Disease Classification](https://arxiv.org/abs/2106.05915) | 2021 | NIH-CXR14; CheXpert; (MIMIC-CXR also reported) | DenseNet-121 + anatomy-aware attention (CNN) | Macro AUCs: NIH **0.8578**; CheXpert **0.9207**; MIMIC-CXR **0.8404** |


- [CheXNet](https://arxiv.org/pdf/1711.05225)
- [CheXNeXt](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686)
- [Other papers with same database](https://paperswithcode.com/dataset/chestx-ray8) - No longer works, it now redirects to their github

## VLMs Vision-language models

<img width="521" height="160" alt="image" src="https://github.com/user-attachments/assets/1a5b6d1c-dd60-4edc-83f5-7b0e09033261" />

- [LLaVA-OV (OneVision)](https://arxiv.org/abs/2408.03326) – *LLaVA-OneVision: Easy Visual Task Transfer*  
- [LLaVA-Med](https://arxiv.org/abs/2306.00890) – *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day*  
- [RaDialog](https://arxiv.org/abs/2311.18681) – *RaDialog: A Large Vision-Language Model for Radiology Report Generation and Conversational Assistance*  
- [CheXagent](https://arxiv.org/abs/2401.12208) – *CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation*  
    - [Page](https://stanford-aimi.github.io/chexagent.html)  
    - [GitHub](https://github.com/Stanford-AIMI/CheXagent)  
- [MAIRA](https://www.microsoft.com/en-us/research/project/project-maira/) *(Multimodal AI for Radiology Applications)* – Microsoft Health Futures Research  
    - [PadChest-GR](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) Dataset  
    - Chest X-ray (CXR)-specialised multimodal report generation model [MAIRA-2](https://arxiv.org/pdf/2406.04449)  
    - [RAD-DINO](https://arxiv.org/pdf/2401.10815) Biomedical image encoder  
- [RadVLM](https://arxiv.org/abs/2502.03333) – *RadVLM: A Multitask Conversational Vision-Language Model for Radiology*  

## VLMs Classification F1-Score

<img width="648" height="381" alt="image" src="https://github.com/user-attachments/assets/76f15148-f9e3-4094-80e5-3c76888f11d0" />

## Models for Healthcare Available on Azure AI Foundry

- [CXRReportGen](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/healthcare-ai/deploy-cxrreportgen) - grounded report generation model for chest X-rays. This multimodal AI model incorporates current and prior images, along with key patient information, to generate detailed, structured reports from chest X-rays. The reports highlight AI-generated findings based directly on the images, to align with human-in-the-loop workflows.

- [MedImageInsight](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/healthcare-ai/deploy-medimageinsight) - embedding model that enables sophisticated image analysis, including classification and similarity search in medical imaging. Researchers can use the model embeddings in simple zero-shot classifiers. They can also build adapters for their specific tasks, thereby streamlining workflows in radiology, pathology, ophthalmology, dermatology, and other modalities. 

## References of theory
- Attention for Transformers
    - [First attention paper](https://arxiv.org/abs/1409.0473)
    - [First transformer paper](https://arxiv.org/abs/1706.03762)
- Model architectures
  - Convolutional Neural Networks
    - [ResNet](https://arxiv.org/abs/1512.03385)
    - [UNet](https://arxiv.org/abs/1505.04597)
- Learning techniques
  - Transfer learning

## Code

- Kaggle code example [LINK TO EXAMPLE KAGGLE NOTEBOOK](https://www.kaggle.com/code/kmader/train-simple-xray-cnn)
- My Kaggle code [LINK TO KAGGLE NOTEBOOK](https://www.kaggle.com/code/devmuiz/chest/edit)
- My Colab code [https://colab.research.google.com/drive/1umNFkP6SWtUC7W4cUycM4Cd3pHYMwLcw?usp=sharing]

## Things to watch when trainning NN
- Class imbalances
  - Test set and validation
  - Trainning class weights
- Data filtration
  - Same patient ID on same set (Train/Val/Test)

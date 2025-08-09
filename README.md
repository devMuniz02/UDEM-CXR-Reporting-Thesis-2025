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

## Chest X-Ray Datasets

<img width="620" height="277" alt="image" src="https://github.com/user-attachments/assets/c4b2ea25-1c54-4ba4-8057-40bfdbdb6a1a" />

- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.1.0/) - Restricted-access
- [OpenI](https://pubmed.ncbi.nlm.nih.gov/26133894/) - aka ChestX-ray14
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [BraX](https://physionet.org/content/brax/1.1.0/) - Restricted-access
- [CandidPTX](https://ourarchive.otago.ac.nz/esploro/outputs/dataset/CANDID-PTX/9926556140101891) - Closed access to users outside Health New Zealand
- [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC) - 
- [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) -> Grounded-Reporting -> [PadChest-GR](https://bimcv.cipf.es/bimcv-projects/padchest-gr/)
- [VinDR](https://www.physionet.org/content/vindr-cxr/1.0.0/) - Restricted-access
- [MIDRC](https://www.midrc.org/) - Couldn't find the exact link to download
- JF Healthcare - Not available
- [CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus)

### PEF Dataset - NIH and/or CheXpert Plus

- [LINK TO DRIVE](https://drive.google.com/drive/u/0/folders/0AConL2XD0ndHUk9PVA)

## Papers

### [ChestX-ray8](https://arxiv.org/abs/1705.02315)

### [CheXNeXt](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686)

### [CheXNet](https://arxiv.org/pdf/1711.05225)

### [Other papers with same database](https://paperswithcode.com/dataset/chestx-ray8) NO LONGER WORKS, IT NOW REDIRECTS TO THEIR GITHUB

# VLMs Vision-language models

<img width="521" height="160" alt="image" src="https://github.com/user-attachments/assets/1a5b6d1c-dd60-4edc-83f5-7b0e09033261" />

## LLaVA-OV

## LLaVA-Med

## RaDialog

## ChestXAgent

### [LLM ChestXAgent Page](https://stanford-aimi.github.io/chexagent.html)

### [LLM ChestXAgent GitHub](https://github.com/Stanford-AIMI/CheXagent)

## [MAIRA](https://www.microsoft.com/en-us/research/project/project-maira/) *(Multimodal AI for Radiology Applications)*  - Microsoft Health Futures Research 

### [PadChest-GR](https://bimcv.cipf.es/bimcv-projects/padchest-gr/) Database

### Chest X-ray (CXR)-specialised multimodal report generation model [MAIRA-2](https://arxiv.org/pdf/2406.04449)

### [RAD-DINO](https://arxiv.org/pdf/2401.10815) Biomedical image encoder

## RadVLM

## Models for Healthcare Available on Azure AI Foundry

### [CXRReportGen](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/healthcare-ai/deploy-cxrreportgen) - grounded report generation model for chest X-rays. This multimodal AI model incorporates current and prior images, along with key patient information, to generate detailed, structured reports from chest X-rays. The reports highlight AI-generated findings based directly on the images, to align with human-in-the-loop workflows.

### [MedImageInsight](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/healthcare-ai/deploy-medimageinsight) - embedding model that enables sophisticated image analysis, including classification and similarity search in medical imaging. Researchers can use the model embeddings in simple zero-shot classifiers. They can also build adapters for their specific tasks, thereby streamlining workflows in radiology, pathology, ophthalmology, dermatology, and other modalities. 

## Papers Attention for LLM Theory

### [First attention](https://arxiv.org/abs/1409.0473)

### [First transformer](https://arxiv.org/abs/1706.03762)

## Code

- [LINK TO DRIVE](https://drive.google.com/drive/u/0/folders/0AConL2XD0ndHUk9PVA)

### Kaggle code example [LINK TO EXAMPLE KAGGLE NOTEBOOK](https://www.kaggle.com/code/kmader/train-simple-xray-cnn)


### My Kaggle code [LINK TO KAGGLE NOTEBOOK](https://www.kaggle.com/code/devmuiz/chest/edit)

### My Colab code [https://colab.research.google.com/drive/1umNFkP6SWtUC7W4cUycM4Cd3pHYMwLcw?usp=sharing]

## Things to watch when trainning NN
- Class imbalances
  - Test set and validation
  - Trainning class weights
- Data filtration
  - Same patient ID on same set (Train/Val/Test)

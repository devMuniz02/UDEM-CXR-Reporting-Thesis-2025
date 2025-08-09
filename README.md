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
- [VinDR](https://www.physionet.org/content/vindr-cxr/1.0.0/) - Restricted-access
- [MIDRC](https://www.midrc.org/) - Couldn't find the exact link to download
- JF Healthcare - Not available
- [CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus)
    - [Paper](https://arxiv.org/pdf/2405.19538v1)

### Dataset used in PEF - NIH and/or CheXpert Plus

- [LINK TO DRIVE](https://drive.google.com/drive/u/0/folders/0AConL2XD0ndHUk9PVA)

## Papers

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

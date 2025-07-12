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

## Database 

[LINK TO DRIVE](https://drive.google.com/drive/u/0/folders/0AConL2XD0ndHUk9PVA)

[LINK TO DOWNLOAD ORIGINAL NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC)

[LINK TO KAGGLE](https://www.kaggle.com/datasets/nih-chest-xrays/data)

### Original paper [LINK TO PAPER](https://arxiv.org/abs/1705.02315)

### Other papers with same database [LINK](https://paperswithcode.com/dataset/chestx-ray8)

### Other paper [LINK](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002686)

### Other paper [LINK](https://arxiv.org/pdf/1711.05225)

### [LINK](https://arxiv.org/abs/1409.0473)

### Kaggle code example [LINK TO EXAMPLE KAGGLE NOTEBOOK](https://www.kaggle.com/code/kmader/train-simple-xray-cnn)


### My Kaggle code [LINK TO KAGGLE NOTEBOOK](https://www.kaggle.com/code/devmuiz/chest/edit)

### My Colab code [https://colab.research.google.com/drive/1umNFkP6SWtUC7W4cUycM4Cd3pHYMwLcw?usp=sharing]

## Things to watch when trainning NN
- Class imbalances
  - Test set and validation
  - Trainning class weights
- Data filtration
  - Same patient ID on same set (Train/Val/Test)

# Chest X-ray Diagnosis and Automated Radiology Report Generation - UDEM PEF-Thesis (Fall-2025)

## [Project options](https://drive.google.com/file/d/1RG4J-OJZHEcZ6JLumfA0csCI9Gp2dGzf/view?usp=sharing)

![Project Options](assets/ProjectOptions.png)

## [Option chosen](https://drive.google.com/file/d/1Gd5-31rOAuWlL_81S1IRSGc9QBF8aOPb/view?usp=sharing)

![Option B](assets/OptionChosen.png)

## [Simple workflow](https://drive.google.com/file/d/1NbkvJL-v_InbMioPg1pficEe3plhqRi1/view?usp=sharing)

## [Research info](https://github.com/devMuniz02/Chest-X-ray-Diagnosis-Automated-Reporting-using-CNNs-and-LLMs---UDEM-PEF-Thesis-Fall-2025/wiki)

## References of theory
- Attention for Transformers
    - [First attention paper](https://arxiv.org/abs/1409.0473)
    - [First transformer paper](https://arxiv.org/abs/1706.03762)
    - [First GPT model OpenAi](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
    - [GPT-2 model OpenAi](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
      - [GPT-2 Model architecture](https://medium.com/%40hsinhungw/gpt-2-detailed-model-architecture-6b1aad33d16b)
      
        ![GPT-2 Model architecture](assets/GPT2ModelArchitecture.png)
- Model architectures
  - Convolutional Neural Networks
    - [ResNet](https://arxiv.org/abs/1512.03385)
    - [UNet](https://arxiv.org/abs/1505.04597)
- Techniques
  - Transfer learning
  - Visual explanations
      - [Grad-CAM](https://arxiv.org/pdf/1610.02391)
      - [Grad-CAM++](https://arxiv.org/pdf/1710.11063)

## Things to watch when trainning NN
- Class imbalances
  - Test set and validation
  - Trainning class weights
- Data filtration
  - Same patient ID on same set (Train/Val/Test)



# Replication Guide

Instructions to **replicate**, **contribute**, or **extend** experiments using **CheXpertPlus** and **MIMIC-CXR** datasets with **DINOv3-based multimodal models**, trained locally or in the cloud using **Google Vertex AI**.

---

## 📂 Datasets

This project uses two publicly available medical imaging datasets: **CheXpertPlus** and **MIMIC-CXR**, both designed for **chest X-ray research** and suitable for **image + report multimodal training**.

---

### 🩻 1. CheXpertPlus Dataset

1. Go to the [Stanford AIMI CheXpertPlus dataset page](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1).
2. Log in with your institutional or personal account.
3. Download the **CSV metadata** file and all **PNG chunks**.

   > 💡 Right-click the file’s download button to copy the **direct download URL**.
4. Download using **AzCopy** from the terminal:

   ```bash
   mkdir -p Datasets/CheXpertPlus
   azcopy copy "PASTE_FULL_DOWNLOAD_LINK_HERE" "Datasets/CheXpertPlus/" --recursive=false
   ```
5. Extract and organize the dataset as follows:

   ```
   Datasets/
   └── CheXpertPlus/
       ├── PNG/
       │   ├── train/
       │   └── valid/
       └── df_chexpert_plus_240401.csv
   ```

---

### 🩺 2. MIMIC-CXR Dataset

1. Request access to **MIMIC-CXR v2.0.0** via **PhysioNet**:
   [https://physionet.org/content/mimic-cxr-jpg/2.0.0/](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
2. Once credentialed access is granted, download:

   * All **image files** (`.jpg` or `.dcm`)
   * Metadata files: `mimic-cxr-2.0.0-metadata.csv` and `mimic-cxr-2.0.0-chexpert.csv`
3. Arrange the dataset as:

   ```
   Datasets/
   └── MIMIC-CXR/
       ├── images/
       │   ├── p10/
       │   ├── p11/
       │   └── ...
       ├── mimic-cxr-2.0.0-metadata.csv
       └── mimic-cxr-2.0.0-chexpert.csv
   ```

---

## 🤗 Hugging Face Integration

As of **November 7 2025**, all **DINOv3 backbone models** on Hugging Face require **explicit access permissions**.
Follow the steps below to ensure your Vertex AI container or local environment can load DINOv3 weights correctly.

---

### 🔐 Enable Access

1. **Create / Log in to your Hugging Face account**
   [huggingface.co/join](https://huggingface.co/join)

2. **Request access to the DINOv3 collection**
   Visit the [DINOv3 Collection](https://huggingface.co/collections/facebook/dinov3) and click **“Request Access”** for the models you’ll use.

3. **Generate a Read Token**

   * Navigate to **Settings → Access Tokens**
   * Click **“New Token”**
   * Set **Role:** *Read*
   * Copy your token for later use.

---

### ☁️ Using the Token in Google Vertex AI

If training within your Vertex AI Docker container, open:

```python
# File: cloud-trainer/trainer/train.py
import os
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HF_TOKEN_HERE"
```

> ⚠️ **Security Tip:** Avoid hard-coding tokens.
> Instead, inject them when launching your custom job:
>
> ```bash
> --env=HUGGING_FACE_HUB_TOKEN=your_token_here
> ```

---

### 💻 Using the Token Locally

If running the repository locally:

```python
import os
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HF_TOKEN_HERE"
```

Or set it globally in your terminal:

```bash
# Windows
setx HUGGING_FACE_HUB_TOKEN "YOUR_HF_TOKEN_HERE"

# macOS / Linux
export HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"
```

---

## 🚀 Replicate Cloud Training on Google Vertex AI

This section details how to configure and run **custom training jobs** on **Google Vertex AI** using your dataset, Docker image, and compute resources.

---

### 📘 Resources

* **Docs:** [Vertex AI Custom Training Overview](https://cloud.google.com/vertex-ai/docs/training/overview)
* **Tutorials:** [Google Cloud Vertex AI Playlist](https://youtube.com/playlist?list=PLIivdWyY5sqJAyUJbbsc8ZyGLNT4isnuB&si=a1DFZEgxf5GcwL0G)

---

### ⚙️ Setup Guide

#### 1️⃣ Create and Configure Your Google Cloud Project

* Open [Google Cloud Console](https://console.cloud.google.com/).
* Create or select a project.
* Enable:

  * **Vertex AI API**
  * **Artifact Registry API**
  * **Cloud Storage API**

---

#### 2️⃣ Install and Initialize Google Cloud CLI (Windows)

Run the following in **Windows Terminal or PowerShell**:

```bash
# Install the Google Cloud CLI
msiexec.exe /i https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe

# Restart terminal, then initialize
gcloud init
```

During setup:

* Sign in with your Google account
* Choose your project
* Set a default region (e.g., `us-central1`)

To switch projects later:

```bash
gcloud config set project <YOUR_PROJECT_ID>
```

---

#### 3️⃣ Upload Your Dataset to Cloud Storage

Use `gcloud storage` (instead of gsutil):

```bash
# Create bucket
gcloud storage buckets create gs://your-bucket-name --location=us-central1

# Upload dataset
gcloud storage cp -r ./data gs://your-bucket-name/data/
```

---

#### 4️⃣ Enable Artifact Registry & Authenticate Docker

```bash
gcloud services enable artifactregistry.googleapis.com
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

#### 5️⃣ Build and Push Your Custom Trainer Image

```bash
# Build image
docker build -t us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/<IMAGE_NAME>:latest .

# Push image
docker push us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/<IMAGE_NAME>:latest
```

---

#### 6️⃣ Launch a Custom Training Job

From CLI or Cloud Console:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="my-custom-training-job" \
  --container-image-uri=us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/<IMAGE_NAME>:latest \
  --args="--data_dir=gs://your-bucket-name/data,--epochs=10" \
  --machine-type=n1-standard-8 \
  --accelerator-type=NVIDIA_TESLA_T4 \
  --accelerator-count=1
```

---

### 🧩 Notes & Best Practices

* Adjust `--machine-type` and `--accelerator-type` for cost vs speed.
* Monitor jobs in **Vertex AI → Training → Custom Jobs**.
* Check logs in **Cloud Logging**.
* Manage jobs:

  ```bash
  gcloud ai custom-jobs list
  gcloud ai custom-jobs describe <JOB_ID>
  ```

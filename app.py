import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import pipeline

# Load language model (optional - uncomment one)
# pipe = pipeline("text-generation", model="microsoft/Phi-4-mini-instruct", trust_remote_code=True)
# pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)

# Dummy CNN model for multi-label classification
class ChestXRayCNN(nn.Module):
    def __init__(self):
        super(ChestXRayCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 14),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = ChestXRayCNN()
model.eval()

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Optional: Generate detailed report using an LLM
def generate_phi_report(reason, findings_list):
    findings_text = ", ".join(findings_list)
    prompt = f"""You are a radiologist. Write a concise and professional chest X-ray report.
Indication: {reason or "Not specified"}
Findings: {findings_text}
Report:"""
    output = pipe(prompt, max_new_tokens=200, do_sample=True)[0]['generated_text']
    return output.split("Report:")[-1].strip() if "Report:" in output else output.strip()

def predict(image, reason=""):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        preds = model(image).squeeze().numpy()

    findings = [DISEASES[i] for i, p in enumerate(preds) if p > 0.5]

    if findings:
#        return f"""Radiology Report:
#Indication: {reason or "Not specified"}
#Findings: {", ".join(findings)}
#Impression: Further clinical correlation recommended."""
        return generate_phi_report(reason, findings)  # Use this line if LLM is enabled
    else:
        return f"""Radiology Report:
Indication: {reason or "Not specified"}
Findings: No significant abnormal findings.
Impression: Normal chest X-ray.
"""

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Chest X-ray"),
        gr.Textbox(label="Reason for X-ray (optional)", placeholder="e.g. Fever, cough, chest pain")
    ],
    outputs="text",
    title="Chest X-ray Report Generator",
    description="🩻 Uses a CNN for multi-label classification and optionally a language model for report generation. Not for clinical use."
)

iface.launch(server_name="0.0.0.0", server_port=8000)

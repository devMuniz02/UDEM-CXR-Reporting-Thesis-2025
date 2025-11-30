import os
import re
from PIL import UnidentifiedImageError
from torch.utils.data import Dataset
from utils.processing import join_uri, pil_from_path, open_text

import re
# =========================
# Helpers for report sections
# =========================

def normalize_section_names(section_names):
    # first, lower case all
    section_names = [s.lower().strip() for s in section_names]

    frequent_sections = {
        "preamble": "preamble",  # 227885
        "impression": "impression",  # 187759
        "comparison": "comparison",  # 154647
        "indication": "indication",  # 153730
        "findings": "findings",  # 149842
        "examination": "examination",  # 94094
        "technique": "technique",  # 81402
        "history": "history",  # 45624
        "comparisons": "comparison",  # 8686
        "clinical history": "history",  # 7121
        "reason for examination": "indication",  # 5845
        "notification": "notification",  # 5749
        "reason for exam": "indication",  # 4430
        "clinical information": "history",  # 4024
        "exam": "examination",  # 3907
        "clinical indication": "indication",  # 1945
        "conclusion": "impression",  # 1802
        "chest, two views": "findings",  # 1735
        "recommendation(s)": "recommendations",  # 1700
        "type of examination": "examination",  # 1678
        "reference exam": "comparison",  # 347
        "patient history": "history",  # 251
        "addendum": "addendum",  # 183
        "comparison exam": "comparison",  # 163
        "date": "date",  # 108
        "comment": "comment",  # 88
        "findings and impression": "impression",  # 87
        "wet read": "wet read",  # 83
        "comparison film": "comparison",  # 79
        "recommendations": "recommendations",  # 72
        "findings/impression": "impression",  # 47
        "pfi": "history",
        'recommendation': 'recommendations',
        'wetread': 'wet read',
        'ndication': 'impression',  # 1
        'impresson': 'impression',  # 2
        'imprression': 'impression',  # 1
        'imoression': 'impression',  # 1
        'impressoin': 'impression',  # 1
        'imprssion': 'impression',  # 1
        'impresion': 'impression',  # 1
        'imperssion': 'impression',  # 1
        'mpression': 'impression',  # 1
        'impession': 'impression',  # 3
        'findings/ impression': 'impression',  # ,1
        'finding': 'findings',  # ,8
        'findins': 'findings',
        'findindgs': 'findings',  # ,1
        'findgings': 'findings',  # ,1
        'findngs': 'findings',  # ,1
        'findnings': 'findings',  # ,1
        'finidngs': 'findings',  # ,2
        'idication': 'indication',  # ,1
        'reference findings': 'findings',  # ,1
        'comparision': 'comparison',  # ,2
        'comparsion': 'comparison',  # ,1
        'comparrison': 'comparison',  # ,1
        'comparisions': 'comparison'  # ,1
    }

    p_findings = [
        'chest',
        'portable',
        'pa and lateral',
        'lateral and pa',
        'ap and lateral',
        'lateral and ap',
        'frontal and',
        'two views',
        'frontal view',
        'pa view',
        'ap view',
        'one view',
        'lateral view',
        'bone window',
        'frontal upright',
        'frontal semi-upright',
        'ribs',
        'pa and lat'
    ]
    p_findings = re.compile('({})'.format('|'.join(p_findings)))

    main_sections = [
        'impression', 'findings', 'history', 'comparison',
        'addendum'
    ]
    for i, s in enumerate(section_names):
        if s in frequent_sections:
            section_names[i] = frequent_sections[s]
            continue

        main_flag = False
        for m in main_sections:
            if m in s:
                section_names[i] = m
                main_flag = True
                break
        if main_flag:
            continue

        m = p_findings.search(s)
        if m is not None:
            section_names[i] = 'conclusion'

    return section_names


def section_text(text):
    """Splits text into sections and normalizes section names."""
    p_section = re.compile(r'\n ([A-Z ()/,-]+):\s', re.DOTALL)

    sections = []
    section_names = []
    section_idx = []

    idx = 0
    s = p_section.search(text, idx)

    if s:
        sections.append(text[0:s.start(1)])
        section_names.append('preamble')
        section_idx.append(0)

        while s:
            current_section = s.group(1).lower()
            # get the start of the text for this section
            idx_start = s.end()
            # skip past the first newline to avoid some bad parses
            idx_skip = text[idx_start:].find('\n')
            if idx_skip == -1:
                idx_skip = 0

            s = p_section.search(text, idx_start + idx_skip)

            if s is None:
                idx_end = len(text)
            else:
                idx_end = s.start()

            sections.append(text[idx_start:idx_end])
            section_names.append(current_section)
            section_idx.append(idx_start)
    else:
        sections.append(text)
        section_names.append('full report')
        section_idx.append(0)

    section_names = normalize_section_names(section_names)

    # remove empty findings/impression sections
    for i in reversed(range(len(section_names))):
        if section_names[i] in ('impression', 'findings'):
            if sections[i].strip() == '':
                sections.pop(i)
                section_names.pop(i)
                section_idx.pop(i)

    # if there is no impression/findings, treat last paragraph as a section
    if ('impression' not in section_names) and ('findings' not in section_names):
        if '\n \n' in sections[-1]:
            sections.append('\n \n'.join(sections[-1].split('\n \n')[1:]))
            sections[-2] = sections[-2].split('\n \n')[0]
            section_names.append('last_paragraph')
            section_idx.append(section_idx[-1] + len(sections[-2]))

    return sections, section_names, section_idx

def extract_findings(report_text: str) -> str:
    """
    Extract the FINDINGS section from a radiology report using section_text().
    If no FINDINGS section is found, return the whole report text
    (whitespace-normalized).
    """
    # Use your structured section parser
    sections, section_names, section_idx = section_text(report_text)

    findings_text = None
    for sec, name in zip(sections, section_names):
        if name == "findings":
            findings_text = sec
            break

    if findings_text is None:
        for sec, name in zip(sections, section_names):
            if name in ["impression", "conclusion", "last_paragraph"]:
                findings_text = sec
                break
        # Fallback: use full report
        findings_text = report_text

    # Clean up whitespace
    findings_text = re.sub(r"\s+", " ", findings_text).strip()
    return findings_text

# def extract_findings(report_text: str) -> str:
#     """
#     Extract the FINDINGS section from a radiology report.
#     If not found, return the whole report text.
#     """
#     match = re.search(r"FINDINGS:\s*(.*?)\s*(IMPRESSION:|$)", report_text, re.S | re.I)
#     findings = match.group(1).strip() if match else report_text.strip()
#     findings = re.sub(r"\s+", " ", findings).strip()  # Clean up whitespace
#     return findings


# class MIMICDataset(Dataset):
#     """
#     Expects a DataFrame with at least a 'path' (relative DICOM path) and
#     you provide:
#       - images_dir: base dir (local or gs://) where your pre-rendered JPG/PNG live
#       - reports_dir: base dir (local or gs://) where report TXT files live
#     It will map '.../dXXXXX.dcm' -> '.../dXXXXX.jpg' (adjust if needed).
#     """
#     def __init__(self, dataframe, images_dir: str, reports_dir: str, transform=None):
#         self.df = dataframe.reset_index(drop=True)
#         self.images_dir = images_dir  # can be local or gs://
#         self.reports_dir = reports_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         rel_path = str(row.get("path", ""))  # e.g. p10/p100XXXX/sXXXXX/dXXXXX.dcm
#         if not rel_path:
#             raise ValueError("Row is missing 'path' column required to locate files.")

#         # Image path: swap extension .dcm -> .jpg (or .png) according to your preprocessing
#         image_name = os.path.basename(rel_path).replace(".dcm", ".jpg")
#         image_path = join_uri(self.images_dir, image_name)

#         # Report path: derive folder path and append .txt
#         rel_dir = os.path.dirname(rel_path)  # p10/p100XXXX/sXXXXX
#         report_path = join_uri(self.reports_dir, f"{rel_dir}.txt")

#         # Load image
#         try:
#             im = pil_from_path(image_path)
#         except (FileNotFoundError, UnidentifiedImageError):
#             print(f"[WARN] Problem with image: {image_path}, skipping index {idx}")
            
#             # Evitar recursión infinita si hay muchos archivos faltantes
#             next_idx = (idx + 1) % len(self.df)
#             if next_idx == idx:
#                 # Solo quedaba este elemento y también falta → sí lanzamos error
#                 raise RuntimeError("No valid images found in dataset.")
            
#             return self.__getitem__(next_idx)
        
#         image = self.transform(im) if self.transform else im

#         # Load & clean report text (best-effort)
#         findings = ""
#         try:
#             with open_text(report_path, encoding="utf-8") as f:
#                 full_report = f.read()
#             # Robust FINDINGS extraction; fallback to whole report
#             findings = extract_findings(full_report)
#         except FileNotFoundError:
#             findings = ""

#         return image, findings, image_path, report_path

class MIMICDataset(Dataset):
    """
    Expects a DataFrame with at least a 'path' (relative DICOM path) and
    you provide:
      - images_dir: base dir (local or gs://) where your pre-rendered JPG/PNG live
      - reports_dir: base dir (local or gs://) where report TXT files live
    It will map '.../dXXXXX.dcm' -> '.../dXXXXX.jpg' (adjust if needed).
    """
    def __init__(self, dataframe, images_dir: str, reports_dir: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_dir  # can be local or gs://
        self.reports_dir = reports_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = str(row.get("path", ""))  # e.g. p10/p100XXXX/sXXXXX/dXXXXX.dcm
        if not rel_path:
            raise ValueError("Row is missing 'path' column required to locate files.")

        # Image path: swap extension .dcm -> .jpg (or .png) according to your preprocessing
        image_name = os.path.basename(rel_path).replace(".dcm", "")
        # image_path = join_uri(self.images_dir, image_name)
        image_dir = os.path.dirname(rel_path)[10:]
        image_path = join_uri(self.images_dir, f"{image_dir}/{image_name}.png")

        # Report path: derive folder path and append .txt
        rel_dir = os.path.dirname(rel_path)  # p10/p100XXXX/sXXXXX
        report_path = join_uri(self.reports_dir, f"{rel_dir}.txt")

        # Load image
        # Load image
        try:
            im = pil_from_path(image_path)
        except FileNotFoundError:
            print(f"[WARN] Image file not found: {image_path}, skipping index {idx}")
            
            # Evitar recursión infinita si hay muchos archivos faltantes
            next_idx = (idx + 1) % len(self.df)
            if next_idx == idx:
                # Solo quedaba este elemento y también falta → sí lanzamos error
                raise FileNotFoundError(f"No valid images found in dataset.")
            
            return self.__getitem__(next_idx)
        image = self.transform(im) if self.transform else im

        # Load & clean report text (best-effort)
        findings = ""
        try:
            with open_text(report_path, encoding="utf-8") as f:
                full_report = f.read()
            # Robust FINDINGS extraction; fallback to whole report
            findings = extract_findings(full_report)
        except FileNotFoundError:
            findings = ""

        return image, findings, image_path, report_path
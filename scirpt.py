import pdfplumber
import re
import os
import argparse
import torch
import shutil
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def extract_information_from_cv(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        cv_text = ""
        for page in pdf.pages:
            cv_text += page.extract_text()
            
    summary_pattern = re.compile(r'Summary\n(.+?)\n', re.DOTALL)
    summary_match = summary_pattern.search(cv_text)
    summary = summary_match.group(1) if summary_match else None
    
    experience_pattern = re.compile(r'Experience\n(.+?)\nEducation', re.DOTALL)
    experience_match = experience_pattern.search(cv_text)
    experience = experience_match.group(1) if experience_match else None
    
    education_pattern = re.compile(r'Education\n(.+?)\nSkills', re.DOTALL)
    education_match = education_pattern.search(cv_text)
    education = education_match.group(1) if education_match else None
    
    skills_pattern = re.compile(r'Skills\n(.+)', re.DOTALL)
    skills_match = skills_pattern.search(cv_text)
    skills = skills_match.group(1) if skills_match else None
    
    if experience is None:
        experience = "experience"
    if education is None:
        education = "education"
    
    data = experience + "\n" + education
    return data 

def dump_rows(data):
    titles = [
        ["cv-name", "domain","path from"],
    ]
    data = titles + data
    
    

    path = "output"
    filename = os.path.join(path,"resume.csv")
    if os.path.exists(path):
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
    else:
        os.makedirs(filename)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data) 
     




def predict(path):
    file_list  = os.listdir(path)
    row_data = []
    for cv in file_list:
        cv_name = cv
        cv_path = os.path.join(path,cv)
        data = extract_information_from_cv(cv_path)
        # data_list = [dir_name, data]
        # row_data.append(data_list)
        # print(f"Finish {dir_name} domain")

        inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
        classes = ['ACCOUNTANT',
        'ADVOCATE',
        'AGRICULTURE',
        'APPAREL',
        'ARTS',
        'AUTOMOBILE',
        'AVIATION',
        'BANKING',
        'BPO',
        'BUSINESS-DEVELOPMENT',
        'CHEF',
        'CONSTRUCTION',
        'CONSULTANT',
        'DESIGNER',
        'DIGITAL-MEDIA',
        'ENGINEERING',
        'FINANCE',
        'FITNESS',
        'HEALTHCARE',
        'HR',
        'INFORMATION-TECHNOLOGY',
        'PUBLIC-RELATIONS',
        'SALES',
        'TEACHER']
        predicted = classes[predicted_class]
        data_list = [cv_name, predicted,cv_path]
        row_data.append(data_list)

    return row_data
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict domains from path of CVs")
    parser.add_argument("--path", type=str, required=True, help="Path to the directory containing CVs")
    args = parser.parse_args()

    path = args.path

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=24)
    tokenizer = AutoTokenizer.from_pretrained("./resume_transformer/")
    loaded_model = AutoModelForSequenceClassification.from_pretrained('./resume_transformer/')

    r_list = predict(path)
    for i in r_list:
        path = os.path.join("output",i[1])
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.move(i[2],path)
        print(f"The Domain of {i[0]} is {i[1]}")
    dump_rows(r_list)
    
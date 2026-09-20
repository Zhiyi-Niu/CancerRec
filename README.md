# CancerRec: Multiple Urological Cancer Recognition


## 📁 Project Structure

```Plain Text
CancerRec/
├── Cancer_Rec_Code/
│   ├── Cancer_Data/                  # Empty placeholder folder for running dataset
│   ├── cancer_model.py
│   ├── cancer_pipeline.py
│   ├── config.yaml
│   ├── explicit_response_features.py
│   ├── requirements.txt
│   ├── run_cancer_recognition.py
│   ├── run_main.sh
│   ├── stmap_generation.py
│   └── verify_cancer_recognition.py
├── Fig Source Data/                  # Raw original data for paper figure plotting
├── README.md
└── Supplementary_Video_1.mp4         # Project demonstration video
```

## 🎬 Demo Video
[<video src="Supplementary_Video_1.mp4" controls width="500"></video>](https://github.com/user-attachments/assets/bb37f1da-4e99-4d7a-81ac-fd603a4ecaf7)

## ⚙️ Environment Requirements

All Python dependencies are listed in `Cancer_Rec_Code/requirements.txt`\.

```Plain Text
cd Cancer_Rec_Code
pip install -r requirements.txt
```

## 📊 Running Dataset

`Cancer_Rec_Code/Cancer_Data/` is an empty placeholder folder\. Due to the large size of the original experimental data, it is not uploaded to this repository\.

**Full dataset download \(Baidu Drive\):**

[https://drive\.google\.com/xxxxxx](https://pan.baidu.com/s/1XsQMErixk3KECwjI1y88vQ?pwd=2026)

After downloading, place all data into `Cancer_Rec_Code/Cancer_Data/` and modify the data path in`config.yaml` before running\.

## 📈 Figure Source Data \(For Paper Plotting\)

The `Fig Source Data/` folder stores **raw original data for reproducing all paper figures**\.

This part of data is **independent** from the model running dataset above, specifically for result visualization and figure generation\.

## 🚀 Quick Start

### 1\. Run the full prediction pipeline

```Plain Text
cd Cancer_Rec_Code
bash run_main.sh
```

### 2\. Run main Python script

```Plain Text
cd Cancer_Rec_Code
python run_cancer_recognition.py
```


## 💡 Notes

- The running dataset is stored on Google Drive due to file size limitations

- `Fig Source Data` is specially used for paper figure reproduction

# Telco Churn Prediction Service and Uplift Modeling

This project implements an **End-to-End Machine Learning Service** for predicting customer churn in a telecommunications setting. The goal is not just to train a churn prediction model, but also to build a deployable system that can identify at-risk customers and support targeted retention strategies through **uplift modeling**.

The core of this repository is a **FastAPI** service that exposes two trained models: an **XGBoost** classifier and a **Multi-Layer Perceptron (MLP)**. Both models are wrapped in **custom scikit-learn pipelines**, allowing them to accept raw input data and produce predictions in a consistent way. This ensures that the same preprocessing logic used during training is also applied at inference time, avoiding training-serving mismatches.

The project combines two complementary components:

- **Churn Prediction** using two supervised models (XGBoost and MLP) trained on historical customer data, including demographics, contract information, and usage behavior. The output is the probability that a customer will churn.

- **Uplift Modeling** on top of churn prediction to estimates the **incremental effect of retention interventions**. Instead of simply identifying high-risk customers, this approach helps identify persuadable customers — those who are likely to stay only if given an incentive. This allows marketing efforts to focus on customers where intervention has measurable impact.


## Key Features
- **Structured Pipeline Architecture**:
    - **ETL Pipeline**: A dedicated ETL workflow for ingesting and cleaning raw data, and creating reproducible raw/processed splits.
    - **End-to-End Inference Pipelines**: Separate, fully-encapsulated inference pipelines for both XGBoost and MLP models. These artifacts take raw data as input and output predictions.
    - **Leakage-Proof Data Flow**: Models are trained on raw splits. By handling all transformations internally within the pipeline, we ensure zero data leakage and total parity between training and inference.
- **Feature Engineering**: Leveraged a mix of built-in scikit-learn transformers and custom Python classes, designed to integrate cleanly with the sklearn API.
- **Production-Oriented Engineering**: 
    - Dockerized deployment using a lightweight Python 3.11 slim, optimized for CPU inference to achieve a lightweight image (< 1GB).
    - Carefully managed requirements.txt to ensure consistency between local development and production.
    - Centralized configuration through a **config.yaml** file for managing paths, hyperparameters, and feature definitions without modifying core code.

## Project Structure

```text
telco-churn-uplift/
├── data/
│   ├── raw/                
│   └── processed/          
├── models/                             # Inference artifacts (ignored by Git, mounted via Docker)
│   ├── preprocessor_pipeline.joblib    # Full ETL pipeline: engineering, transformations, scaling, & encoding
│   ├── xgboost_v1.joblib               # End-to-End Pipeline: Preprocessing + Trained XGBoost Classifier
│   └── mlp_v1.joblib                   # End-to-End Pipeline: Preprocessing + Trained MLP Classifier
├── notebooks/              
├── src/                   
│   ├── etl/                
│   ├── features/           # Custom Sklearn Transformer classes
│   ├── models/             # Training scripts
│   ├── serve/              # FastAPI application and prediction service
│   └── utils/              # Logging and config loader functions
├── tests/                  
├── config.yaml             # Global configuration (paths, parameters, hyperparameters, feature lists)
├── Dockerfile              # Optimized multi-layer build for CPU-only inference
├── .dockerignore           
├── .gitignore              
├── requirements.txt        
├── LICENSE                 
└── README.md 
```

## Usage (Local Setup)
To get the project up and running locally, follow these steps in order.

**1. Environment Setup** 
 
First, clone the repo and set up a virtual environment to keep your global Python installation clean:

```bash
git clone https://github.com/Khaghshenas/telco-churn-uplift.git
cd telco-churn-uplift
python3 -m venv venv  # python -m venv .venv-rag in Windows
source venv/bin/activate # .venv-rag\Scripts\Activate.ps1 in Windows (PowerShell)
pip install -r requirements.txt
```

**2. Data Preparation**

The raw dataset is not included in this repository. To run the pipeline:
- First Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the ```WA_Fn-UseC_-Telco-Customer-Churn.csv``` file in ```data/raw/```.
- Then Run the ETL script to generate train/test splits in ```data/processed/```:
```bash
python src/etl/churn_etl.py
``` 

**3. Training & Inference**

Once your data is ready, you have a few ways to interact with the models:

- **Retrain models**: If you want to experiment with different architectures, hyperparameters or updated data, you can trigger the training pipelines for either model. These scripts will automatically save the new end-to-end ```.joblib``` artifacts to the ```models/``` directory. 
```bash
python src/models/train_xgboost.py
python src/models/train_mlp.py
```
- **CLI Inference**: To test the pipeline without starting the web service, you can run the prediction script directly. This is the fastest way to verify that your custom transformers and paths are working correctly.
```bash
python src/serve/predict.py
```
- **Start the FastAPI Service**: For production-style testing, run the API. You can use ```--reload``` so that any changes you make to the source code are reflected immediately.
```bash
uvicorn src.serve.api:app --reload --port 8000
```
Once the service is running, open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
 in your browser. The Swagger UI lets you send test JSON payloads to the models and view churn probabilities in real time:
 
 ![API Prediction Demo: Request Body](docs/images/api_request_body.png?raw=true)

 ![Response Body](docs/images/api_response_body.png?raw=true)

## Evaluation Results
## Churn Prediction

The churn prediction model was evaluated on a test set using probability-based and classification metrics.

- **ROC-AUC:** **0.82**, indicating strong discrimination between churners and non-churners.
- **Accuracy:** 79% on the test set.
- **Churn Precision:** 64%, meaning that nearly two-thirds of customers predicted as churners actually churned.
- **Churn Recall:** 51%, capturing over half of true churn cases.

### Confusion Matrix (Test Set)

|               | Predicted No Churn | Predicted Churn |
|---------------|-------------------|-----------------|
| **Actual No Churn** | 927 | 106 |
| **Actual Churn**    | 185 | 189 |


These results show  a reasonable trade-off between false positives and false negatives. The predicted churn probabilities are later used as input for uplift modeling to prioritize targeted retention strategies.

## Uplift Modeling
### Uplift Model Evaluation

The uplift model predicts the incremental effect of retention actions for each customer. Sample uplift scores:

```bash
[ 0.023 0.062 -0.002 -0.033 0.354 0.126 -0.017 -0.059 -0.134 0.015 ]
```

Evaluation on the test set:

- **Uplift @ top 20%:** 0.024 — targeting the top 20% predicted customers yields a small incremental benefit over random selection.  

> **Note:** These results are for demonstration purposes. Using real treatment data and model tuning is expected to significantly improve uplift performance.

## License

This project is licensed under the MIT License.


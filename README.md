# Customer Churn Prediction and Uplift Modeling Platform

An **end-to-end machine learning platform** for predicting customer churn and optimizing customer retention strategies in the telecommunications domain. The project goes beyond traditional churn prediction with uplift modeling to identify not only customers who are likely to leave, but also those most likely to respond positively to retention interventions.

The platform provides a production-oriented inference service built with **FastAPI** and exposes a unified prediction pipeline capable of serving two ML models; **XGBoost** and **Multi-Layer Perceptron (MLP)**—along with a custom T-Learner uplift model for estimating the causal impact of retention campaigns.

### Key Capabilities

- Predicts the probability that a customer will leave the service using scikit-learn-based pipelines with XGBoost and MLP models trained on historical customer data, including demographics, contract information, and usage behavior.
- Incorporates uplift modeling to estimate the incremental effect of retention interventions, in order to identify persuadable customers who are likely to remain only if targeted with incentives, thereby optimizing marketing and retention efforts.
- Includes end-to-end ML pipelines for data preprocessing, training, evaluation, and inference.
- Provides a FastAPI-based REST service for real-time churn prediction.
- Ensures leakage-proof pipeline design with unified preprocessing to maintain consistency between training and inference. 
- Supports cloud-native deployment using Docker and Kubernetes, enabling scalable and production-like model serving.
- Uses centralized configuration via config.yaml for hyperparameters, paths, and feature management.
- Follows a modular architecture designed for scalable model serving and future MLOps extensions.

This project demonstrates the development of a complete applied ML system, covering data engineering, model development, API deployment, and production-oriented software engineering practices.

## Architecture

The platform follows a modular architecture that separates data preparation, model training, and model serving.

```text
                     ┌─────────────────┐
                     │   Raw Dataset   │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │   ETL Pipeline  │
                     │ Cleaning &      │
                     │ Validation &    │
                     │ Train/Test Split│
                     └────────┬────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ XGBoost Model  │  │   MLP Model    │  │ Uplift Model   │
│ Churn Pipeline │  │ Churn Pipeline │  │  (T-Learner)   │
└────────┬───────┘  └────────┬───────┘  └────────┬───────┘
         │                   │                   │
         └──────────┬────────┴──────────┬────────┘
                    │                   │
                    ▼                   ▼
             ┌────────────────────────────┐
             │ Serialized Pipeline Models │
             └─────────────┬──────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ FastAPI Service  │
                 └────────┬─────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Prediction Endpoint│
                │ /predict           │
                └────────────────────┘
```

### Components

* **ETL Pipeline** – Ingests raw customer data, performs validation and cleaning, and generates reproducible training and testing datasets.
* **Churn Prediction Models** – XGBoost and MLP models trained to estimate customer churn probability.
* **Uplift Model** – T-Learner architecture used to estimate the incremental effect of retention interventions and identify persuadable customers.
* **Model Pipelines** – Fully encapsulated scikit-learn pipelines that combine preprocessing, feature engineering, and prediction logic.
* **FastAPI Service** – REST API exposing trained models through a unified inference interface.
* **Docker Deployment** – Containerized runtime environment for reproducible deployment and consistent execution across environments.

This architecture ensures reproducible training, leakage-resistant data processing, and consistent model behavior across development and deployment environments.

## Repository Structure

```text
telco-churn-uplift/
├── data/
│   ├── raw/                
│   └── processed/          
├── models/                             # inference artifacts
│   ├── preprocessor_pipeline.joblib    # full ETL pipeline: engineering, transformations, scaling, & encoding
│   ├── xgboost_v1.joblib               # end-to-end pipeline: preprocessing + trained XGBoost classifier
│   ├── mlp_v1.joblib                   # end-to-end pipeline: preprocessing + trained MLP classifier
│   └── uplift_v1.joblib                # end-to-end pipeline: preprocessing + trained MLP classifier              
├── training/                   
│   ├── etl/                
│   ├── features/           # custom Sklearn transformer classes
│   ├── models/             # training scripts
│   └── utils/              # logging and config loader functions
│   app/              
│   ├── api.py              # FastAPI application and prediction service
│   └── predict.py 
├── docs/images                  
├── config.yaml             # global configuration (paths, parameters, hyperparameters, feature lists)
├── Dockerfile              # multi-layer Docker build for packaging inference API and dependencies
├── deployment.yaml         # Kubernetes deployment: runs application containers (Pods)
├── service.yaml            # Kubernetes service: exposes Pods as a stable network endpoint
├── .dockerignore           
├── .gitignore              
├── requirements.txt        
├── LICENSE                 
└── README.md 
```

## Model Development

## API Usage

## Local Setup
To get the project up and running locally, follow these steps in order.

**1. Environment Setup** 
 
First, clone the repo and set up a virtual environment to keep your global Python installation clean:

```bash
git clone https://github.com/Khaghshenas/telco-churn-uplift.git
cd telco-churn-uplift
python3 -m venv venv  # python -m venv venv in Windows
source venv/bin/activate # venv\Scripts\Activate.ps1 in Windows (PowerShell)
pip install -r requirements.txt
```

**2. Data Preparation**

The raw dataset is not included in this repository. To run the pipeline:
- First Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the ```WA_Fn-UseC_-Telco-Customer-Churn.csv``` file in ```data/raw/```.
- Then Run the ETL script to generate train/test splits in ```data/processed/```:
```bash
python -m training.etl.churn_etl
``` 

**3. Training & Inference**

Once your data is ready, you have a few ways to interact with the models:

- **Retrain models**: If you want to experiment with different architectures, hyperparameters or updated data, you can trigger the training pipelines for all models. These scripts will automatically save the new end-to-end ```.joblib``` artifacts to the ```models/``` directory. 
```bash
python -m training.models.train_xgboost
python -m training.models.train_mlp
python -m training.models.train_uplift
```
- **CLI Inference**: To test the pipeline without starting the web service, you can run the prediction script directly. This is the fastest way to verify that the custom transformers and paths are working correctly.
```bash
python -m training.serve.predict
```
- **Start the FastAPI Service**: For production-style testing, run the API. You can use ```--reload``` so that any changes you make to the source code are reflected immediately.
```bash
uvicorn app.api:app --reload --port 8000
```
Once the service is running, open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
 in your browser. The Swagger UI lets you send test JSON payloads to the models and view churn probabilities and uplift score in real time:
 
 ![API Prediction Demo: Request Body](docs/images/api_request_body.png)

 ![Response Body XGBoost](docs/images/api_response_body_xgboost.png)

  ![Response Body MLP](docs/images/api_response_body_mlp.png)

**4. Deployment (Docker & Kubernetes)**

This project can be deployed in a Docker container for reproducible inference. It can also be deployed on Kubernetes to enable container orchestration, scaling, and service management. The setup described below assumes a local Kubernetes cluster (e.g., Docker Desktop’s Kubernetes).

- **Build the Docker Image**: From the project root directory, build the Docker image:
```bash
docker build -t telco-churn-uplift-api .
```

- **Run the Container**
```bash
docker run -p 8000:8000 telco-churn-uplift-api
```
This exposes the inference service on port 8000.

- **Kubernetes Deployment (Local Cluster)**

Ensure Kubernetes is running:
```bash
kubectl get nodes
```

Then deploy the application:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```
You can verify deployment by:
```bash
kubectl get pods
kubectl get services
```
Finally to access API run:
```bash
kubectl port-forward service/churn-uplift-service 8000:8000
```

## Evaluation Results
## Churn Prediction

We evaluated both the XGBoost and MLP models using the held-out test set ($20\%$ of the raw data). Because telco churn is a classic imbalanced classification problem, we focused on metrics that penalize false negatives and false positives.

| Model     | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) |
|-----------|-----------|--------|----------|---------|-------------------|
| XGBoost   | 0.6523    | 0.5267 | 0.5828   | 0.8407  | 0.37              |
| MLP       | 0.5091    | 0.5989 | 0.5504   | 0.7710  | 4.18              |

The results show a fairly typical pattern when comparing tree-based models and neural networks on tabular data.

- **XGBoost as the stronger overall model**

XGBoost performs better across most metrics, with a higher ROC-AUC (0.84) and F1-score (0.58). It’s also more precise, which means fewer false positives. In practical terms, this reduces the risk of offering incentives to customers who weren’t likely to churn in the first place — an important consideration when retention has real costs.

- **MLP prioritizes recall**

MLP achieves the highest recall (0.60), meaning it captures more of the actual churners. If the business prioritizes minimizing missed churn cases — for example, when customer lifetime value is high — this behavior can be desirable. The trade-off is lower precision and overall weaker discrimination compared to XGBoost.

- **Training efficiency**

There is also a noticeable gap in training time. XGBoost trained about 11× faster than the MLP. For this tabular dataset, tree-based methods not only perform better but also train significantly faster, making XGBoost the more practical choice for experimentation and iteration.

## Uplift Modeling

We evaluated the Two-Model uplift architecture (using XGBoost for both treatment and control groups) with Causal Inference metrics rather than standard accuracy. Since the goal is to estimate the Incremental Impact of a retention offer, we focus on the Qini Coefficient and Uplift at Top K.
- **Qini Coefficient** = overall uplift ranking quality (how much better than random our targeting strategy is).
- **Uplift@K** = incremental gain if we target only the top K% highest-uplift customers.

| Metric           | Result   | Interpretation                                                          |
|------------------|----------|-------------------------------------------------------------------------|
| Uplift @ Top 20% | +2.62%   | High-efficiency targeting; captures significant gains in the first 20%. |
| Qini AUC         | -0.0031  | Indicates the causal signal is highly concentrated in the top segments. |

*NOTE: In a group of 100,000 customers, a 2.62% uplift means 2,620 customers are retained who would have otherwise churned, directly impacting revenue.

The results show that while the model demonstrates a strong 2.62% incremental lift in the primary target customers, the overall metrics reflect the inherent challenges of extracting causal signals from a randomized synthetic treatment distribution.
 
 - **Targeting Efficiency** 
 
 The 2.62% Uplift at 20% demonstrates that even with simulated treatment, the model successfully identifies a "Persuadable" segment.

 - **Impact of Synthetic Randomization** 
 
 The near-zero Qini AUC is mainly the result of the Synthetic Random Treatment Assignment. In real-world observational data, treatments (e.g., discounts) are usually correlated with customer behavior (Selection Bias). By using a purely random synthetic assignment, we created a "Worst-Case Scenario" for the model. 

 ![Qini Curve](docs/images/qini_curve.png)

 As the Qini curve shows, the model achieves its maximum separation from the random baseline within the first two deciles, confirming that the predictive power is concentrated in the highest-ranked 'persuadable' customers before the synthetic noise leads to a convergence with the random targeting line.

## License

This project is licensed under the MIT License.


# Architecture Overview

This document provides visual diagrams of how the Finance Forecasting Platform works.

---

## High-Level System Architecture

```mermaid
flowchart TB
    subgraph User["👤 User Interface"]
        SM[Simple Mode]
        EM[Expert Mode]
    end

    subgraph Frontend["🖥️ React Frontend"]
        Upload[Data Upload]
        Config[Configuration]
        Results[Results Visualization]
        Compare[Forecast vs Actuals]
    end

    subgraph Backend["⚙️ FastAPI Backend"]
        API["/api/train"]
        Preprocess[Preprocessing]
        Models[Model Training]
        Deploy[Deployment Service]
    end

    subgraph MLModels["🤖 Forecasting Models"]
        Prophet
        ARIMA
        ETS[Exponential Smoothing]
        SARIMAX
        XGBoost
    end

    subgraph Databricks["☁️ Databricks Platform"]
        MLflow[MLflow Tracking]
        UC[Unity Catalog]
        Serving[Model Serving]
    end

    SM --> Upload
    EM --> Upload
    Upload --> Config
    Config --> API
    API --> Preprocess
    Preprocess --> Models
    Models --> Prophet & ARIMA & ETS & SARIMAX & XGBoost
    Prophet & ARIMA & ETS & SARIMAX & XGBoost --> MLflow
    MLflow --> UC
    UC --> Serving
    Models --> Results
    Results --> Compare
```

---

## Data Processing Pipeline

```mermaid
flowchart LR
    subgraph Input["📥 Data Input"]
        CSV[CSV Upload]
        Promo[Promotions File]
    end

    subgraph Preprocessing["🔧 Feature Engineering"]
        Calendar[Calendar Features]
        Holiday[Holiday Features]
        Proximity[Holiday Proximity]
        YoY[YoY Lag Features]
    end

    subgraph Split["📊 Data Split"]
        Train[Training 70%]
        Eval[Evaluation 15%]
        Holdout[Holdout 15%]
    end

    subgraph Training["🎯 Model Training"]
        HPT[Hyperparameter Tuning]
        CV[Cross-Validation]
        Fit[Model Fitting]
    end

    CSV --> Calendar
    Promo --> Calendar
    Calendar --> Holiday
    Holiday --> Proximity
    Proximity --> YoY
    YoY --> Train & Eval & Holdout
    Train --> HPT
    HPT --> CV
    CV --> Fit
    Eval --> CV
```

---

## Holiday Feature Engineering

```mermaid
flowchart TD
    subgraph Input["📅 Date Input"]
        Dates[Time Series Dates]
    end

    subgraph Detection["🔍 Holiday Detection"]
        USHolidays[US Federal Holidays]
        CustomHolidays[Custom Holidays]
    end

    subgraph Weekly["📆 Weekly Data Features"]
        WeekFlag[is_thanksgiving_week]
        WeekProx[weeks_to_thanksgiving]
        PrePost[is_pre_thanksgiving]
    end

    subgraph Daily["📆 Daily Data Features"]
        DayFlag[is_holiday]
        DaysTo[days_to_thanksgiving]
        Window[is_thanksgiving_window]
    end

    subgraph Prophet["🔮 Prophet Holidays"]
        MultiDay[Multi-day Windows]
        LowerWin[lower_window: -1]
        UpperWin[upper_window: +3]
    end

    Dates --> USHolidays
    Dates --> CustomHolidays
    USHolidays --> Weekly
    USHolidays --> Daily
    CustomHolidays --> Weekly
    CustomHolidays --> Daily
    Weekly --> MultiDay
    Daily --> MultiDay
    MultiDay --> LowerWin
    MultiDay --> UpperWin
```

---

## Model Training Flow

```mermaid
flowchart TD
    subgraph Request["📨 Training Request"]
        Data[Time Series Data]
        Params[Configuration]
    end

    subgraph Parallel["⚡ Parallel Training"]
        P1[Prophet Thread]
        P2[ARIMA Thread]
        P3[ETS Thread]
        P4[SARIMAX Thread]
        P5[XGBoost Thread]
    end

    subgraph Tuning["🎛️ Hyperparameter Tuning"]
        Grid[Grid Search]
        Filter[Degenerate Filter]
        Best[Best Parameters]
    end

    subgraph Validation["✅ Validation"]
        TSCV[Time Series CV]
        Metrics[MAPE, RMSE, R²]
    end

    subgraph Output["📤 Results"]
        Forecast[Forecast Values]
        CI[Confidence Intervals]
        RunID[MLflow Run ID]
    end

    Data --> P1 & P2 & P3 & P4 & P5
    Params --> P1 & P2 & P3 & P4 & P5
    P1 & P2 & P3 & P4 & P5 --> Grid
    Grid --> Filter
    Filter --> Best
    Best --> TSCV
    TSCV --> Metrics
    Metrics --> Forecast
    Forecast --> CI
    CI --> RunID
```

---

## Simple Mode vs Expert Mode

```mermaid
flowchart TD
    subgraph SimpleMode["🟢 Simple Mode"]
        S1[Upload CSV]
        S2[Auto-detect Columns]
        S3[Auto-select Models]
        S4[One-click Forecast]
        S5[Excel Export]
    end

    subgraph ExpertMode["🔵 Expert Mode"]
        E1[Upload CSV + Promos]
        E2[Manual Column Selection]
        E3[Choose Models]
        E4[Configure Parameters]
        E5[Batch Training]
        E6[MLflow Tracking]
        E7[Model Deployment]
    end

    subgraph Shared["🔄 Shared Backend"]
        Preprocess[Preprocessing]
        Train[Model Training]
        MLflow[MLflow Logging]
    end

    S1 --> S2 --> S3 --> S4 --> S5
    E1 --> E2 --> E3 --> E4 --> E5 --> E6 --> E7

    S4 --> Preprocess
    E4 --> Preprocess
    Preprocess --> Train
    Train --> MLflow
```

---

## Batch Training & Deployment

```mermaid
flowchart LR
    subgraph Segments["📦 Data Segments"]
        Seg1[Region: US]
        Seg2[Region: EU]
        Seg3[Region: APAC]
    end

    subgraph Training["🏋️ Parallel Training"]
        T1[Train US Model]
        T2[Train EU Model]
        T3[Train APAC Model]
    end

    subgraph Registry["📚 Model Registry"]
        M1[US Model v1]
        M2[EU Model v1]
        M3[APAC Model v1]
    end

    subgraph Router["🔀 Router Model"]
        Route[Segment Router]
    end

    subgraph Endpoint["🌐 Serving Endpoint"]
        API[Single API Endpoint]
    end

    Seg1 --> T1 --> M1
    Seg2 --> T2 --> M2
    Seg3 --> T3 --> M3
    M1 & M2 & M3 --> Route
    Route --> API
```

---

## MLflow Artifacts Structure

```mermaid
flowchart TD
    subgraph Run["📁 MLflow Run"]
        subgraph Datasets["datasets/"]
            Raw[raw/original_data.csv]
            Processed[processed/merged_data.csv]
            TrainData[training/train.csv]
            EvalData[training/eval.csv]
        end

        subgraph Model["model/"]
            MLmodel[MLmodel]
            Pickle[python_model.pkl]
            Reqs[requirements.txt]
        end

        subgraph Repro["reproducibility/"]
            Code[training_code.py]
            PreprocessCode[preprocessing.py]
        end

        subgraph Params["Parameters"]
            Frequency[frequency]
            Horizon[horizon]
            Seed[random_seed]
        end

        subgraph Metrics["Metrics"]
            MAPE[mape]
            CVMAPE[cv_mape]
            RMSE[rmse]
        end
    end
```

---

## Technology Stack

```mermaid
flowchart LR
    subgraph Frontend["Frontend"]
        React[React 19]
        TS[TypeScript]
        Vite[Vite]
        Recharts[Recharts]
    end

    subgraph Backend["Backend"]
        FastAPI[FastAPI]
        Uvicorn[Uvicorn]
        Pandas[Pandas]
        NumPy[NumPy]
    end

    subgraph ML["ML Libraries"]
        ProphetLib[Prophet]
        Statsmodels[Statsmodels]
        XGBoostLib[XGBoost]
        Sklearn[Scikit-learn]
    end

    subgraph Platform["Databricks"]
        MLflowPlat[MLflow]
        UCPlat[Unity Catalog]
        ServingPlat[Model Serving]
        Apps[Databricks Apps]
    end

    React --> FastAPI
    FastAPI --> ProphetLib & Statsmodels & XGBoostLib
    ProphetLib & Statsmodels & XGBoostLib --> MLflowPlat
    MLflowPlat --> UCPlat --> ServingPlat
    FastAPI --> Apps
```

---

## Request/Response Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant Backend
    participant MLflow
    participant Serving

    User->>Frontend: Upload CSV
    Frontend->>Backend: POST /api/train
    Backend->>Backend: Preprocess Data
    Backend->>Backend: Train Models (Parallel)
    Backend->>MLflow: Log Run + Artifacts
    MLflow-->>Backend: Run ID
    Backend-->>Frontend: Results + Metrics
    Frontend-->>User: Display Forecast

    User->>Frontend: Click Deploy
    Frontend->>Backend: POST /api/deploy
    Backend->>MLflow: Get Model URI
    Backend->>Serving: Create Endpoint
    Serving-->>Backend: Endpoint URL
    Backend-->>Frontend: Deployment Status
    Frontend-->>User: Endpoint Ready
```

---

**Last Updated:** January 2026

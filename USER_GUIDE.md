# Finance Forecasting Platform - User Guide

> **Your complete guide to creating accurate financial forecasts**

---

## Table of Contents

1. [Quick Start (5 Minutes)](#-quick-start-5-minutes)
2. [Understanding the Modes](#-understanding-the-modes)
3. [Step-by-Step: Simple Mode](#-step-by-step-simple-mode)
4. [Step-by-Step: Expert Mode](#-step-by-step-expert-mode)
5. [Batch Training Multiple Segments](#-batch-training-multiple-segments)
6. [Comparing Forecasts vs Actuals](#-comparing-forecasts-vs-actuals)
7. [Understanding Your Results](#-understanding-your-results)
8. [Data Format Requirements](#-data-format-requirements)
9. [Tips for Better Forecasts](#-tips-for-better-forecasts)
10. [Troubleshooting](#-troubleshooting)

---

## Quick Start (5 Minutes)

### What You Need

| Requirement | Details |
|-------------|---------|
| **Data File** | CSV with dates and values to forecast |
| **Browser** | Chrome, Firefox, Safari, or Edge |
| **Minimum Data** | At least 12 data points (weeks/months) |

### Your First Forecast

```
Step 1: Upload    -->    Step 2: Configure    -->    Step 3: Train    -->    Step 4: View Results
   [CSV]                  [Select columns]          [Click Train]           [See forecast]
```

---

## Understanding the Modes

### Simple Mode (Recommended for Most Users)

| Feature | Description |
|---------|-------------|
| **Best For** | Finance analysts, business users |
| **Setup Time** | < 2 minutes |
| **Configuration** | Automatic (zero-config) |
| **Output** | Excel-friendly results with explanations |

**Simple Mode automatically:**
- Detects your data frequency (daily/weekly/monthly)
- Selects the best models for your data
- Optimizes all parameters
- Provides plain-English explanations

### Expert Mode

| Feature | Description |
|---------|-------------|
| **Best For** | Data scientists, ML engineers |
| **Setup Time** | 5-10 minutes |
| **Configuration** | Full control over all parameters |
| **Output** | Detailed metrics, MLflow tracking |

**Expert Mode gives you:**
- Model selection (Prophet, ARIMA, ETS, SARIMAX, XGBoost)
- Hyperparameter tuning
- Custom covariates
- MLflow experiment tracking

---

## Step-by-Step: Simple Mode

### Step 1: Upload Your Data

1. Click **"Simple Mode"** tab
2. Click **"Upload CSV"** or drag and drop your file
3. Verify the data preview looks correct

**Required columns:**
```
date,revenue
2024-01-01,150000
2024-01-08,165000
2024-01-15,142000
...
```

### Step 2: Verify Auto-Detection

The system will automatically detect:

| Auto-Detected | What It Means |
|---------------|---------------|
| **Date Column** | Which column has your dates |
| **Value Column** | Which column to forecast |
| **Frequency** | Daily, weekly, or monthly |
| **Patterns** | Trend, seasonality, holidays |

Review the "Data Profile" section to confirm detection is correct.

### Step 3: Set Forecast Horizon (Optional)

| Frequency | Default Horizon | Recommendation |
|-----------|-----------------|----------------|
| Daily | 30 days | 7-90 days |
| Weekly | 12 weeks | 4-26 weeks |
| Monthly | 6 months | 3-12 months |

**Tip:** Shorter horizons are more accurate. Start with the default.

### Step 4: Run Forecast

1. Click **"Generate Forecast"**
2. Wait 30 seconds to 2 minutes
3. View your results

### Step 5: Download Results

Click **"Export to Excel"** to get:
- Forecast values with confidence intervals
- Period-by-period breakdown
- Model explanation in plain English
- Audit trail for compliance

---

## Step-by-Step: Expert Mode

### Step 1: Upload Data

1. Click **"Expert Mode"** tab (or stay on default)
2. Click **"Choose File"** and select your CSV
3. Review the data preview

### Step 2: Configure Columns

| Setting | How to Set |
|---------|------------|
| **Time Column** | Select your date column from dropdown |
| **Target Column** | Select the value you want to forecast |
| **Covariates** | (Optional) Select additional predictors |

**Example with covariates:**
```
date,revenue,marketing_spend,is_promotion,temperature
2024-01-01,150000,25000,0,45
2024-01-08,165000,30000,1,42
```

### Step 3: Set Parameters

| Parameter | Options | Recommendation |
|-----------|---------|----------------|
| **Frequency** | daily / weekly / monthly | Match your data |
| **Horizon** | 1-52 periods | Start with 12 |
| **Seasonality Mode** | additive / multiplicative | multiplicative for sales data |

### Step 4: Select Models

| Model | Best For | Supports Covariates |
|-------|----------|---------------------|
| **Prophet** | Most use cases, holidays | Yes |
| **ARIMA** | Stationary data | No |
| **ETS** | Clear trend/seasonality | No |
| **SARIMAX** | Seasonal with covariates | Yes |
| **XGBoost** | Complex patterns | Yes |

**Recommendation:** Select Prophet + one other model for comparison.

### Step 5: Train Models

1. Click **"Train Models"**
2. Monitor progress bar
3. Wait 1-5 minutes depending on data size and models selected

### Step 6: Review Results

After training completes:
- View forecast chart
- Compare model metrics (MAPE, RMSE, R²)
- Download forecast data

---

## Batch Training Multiple Segments

Train forecasts for multiple segments (regions, products, stores) at once.

### Step 1: Prepare Your Data

Your CSV should include segment columns:

```
date,revenue,region,product_category
2024-01-01,150000,US,Electronics
2024-01-01,95000,EU,Electronics
2024-01-01,75000,US,Clothing
...
```

### Step 2: Open Batch Training

1. Complete Steps 1-4 of Expert Mode
2. Click the purple **"Batch Training"** button

### Step 3: Select Segment Columns

1. Check the columns that define your segments
2. Example: Select both "region" and "product_category"
3. Preview shows all segment combinations

```
Example combinations:
  - region=US, product_category=Electronics (156 rows)
  - region=EU, product_category=Electronics (142 rows)
  - region=US, product_category=Clothing (98 rows)
  ...
```

### Step 4: Review & Exclude (Optional)

- Click on any segment to exclude it from training
- Excluded segments appear grayed out
- Use this to skip segments with insufficient data

### Step 5: Train All Segments

1. Click **"Train N Segments"**
2. Monitor overall progress
3. Wait for completion (time depends on segment count)

### Step 6: Review Batch Results

| Column | Meaning |
|--------|---------|
| **Segment** | Combination of segment values |
| **Best Model** | Model with lowest MAPE |
| **MAPE** | Mean Absolute Percentage Error |
| **Status** | Color-coded accuracy indicator |

**Status colors:**
- Green: Excellent (MAPE < 5%)
- Blue: Good (MAPE 5-10%)
- Yellow: Acceptable (MAPE 10-15%)
- Orange: Review (MAPE 15-25%)
- Red: Deviation (MAPE > 25%)

---

## Comparing Forecasts vs Actuals

### Step 1: Generate Forecast First

Complete the training process (Simple or Expert mode).

### Step 2: Upload Actuals Data

1. Click **"Compare with Actuals"** button
2. Upload a CSV with actual values for the forecast period

**Actuals format:**
```
date,actual_value
2024-04-01,158000
2024-04-08,162000
...
```

### Step 3: Map Columns

| Setting | Selection |
|---------|-----------|
| **Date Column** | Select the date column in actuals |
| **Value Column** | Select the actual value column |

### Step 4: View Comparison

The comparison table shows:

| Date | Forecast | Actual | Error | MAPE |
|------|----------|--------|-------|------|
| 2024-04-01 | 155,000 | 158,000 | +3,000 | 1.9% |
| 2024-04-08 | 160,000 | 162,000 | +2,000 | 1.2% |

### Step 5: Filter by Accuracy

Click filter buttons to focus on:
- **Excellent** - MAPE < 5%
- **Good** - MAPE 5-10%
- **Acceptable** - MAPE 10-15%
- **Review** - MAPE 15-25%
- **Deviation** - MAPE > 25%

---

## Understanding Your Results

### Key Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **MAPE** | Average % error | < 10% |
| **RMSE** | Average absolute error | Depends on scale |
| **R²** | How well model fits | > 0.8 |

### MAPE Interpretation for Finance

| MAPE | Rating | Meaning |
|------|--------|---------|
| < 1% | Excellent | Production-ready |
| 1-3% | Very Good | High confidence |
| 3-5% | Good | Reliable for planning |
| 5-10% | Fair | Use with caution |
| > 10% | Review | Investigate data quality |

### Confidence Intervals

The forecast includes upper and lower bounds:

```
Date         Forecast    Lower      Upper
2024-04-01   155,000    145,000    165,000
```

- **80% confidence:** True value likely falls within bounds
- **Wider intervals:** More uncertainty in forecast
- **Use bounds for:** Risk scenarios, budgeting ranges

---

## Data Format Requirements

### Required Structure

Your CSV must have:

| Requirement | Details |
|-------------|---------|
| **Date Column** | Any standard date format |
| **Value Column** | Numeric values to forecast |
| **No Missing Dates** | Fill gaps or use consistent frequency |

### Supported Date Formats

```
2024-01-15           (ISO format - recommended)
01/15/2024           (US format)
15-Jan-2024          (Day-Month-Year)
January 15, 2024     (Full month name)
```

### Optional: Covariates

Add columns that might influence your forecast:

| Covariate Type | Examples |
|----------------|----------|
| **Promotions** | is_black_friday, is_sale_week |
| **Economic** | inflation_rate, interest_rate |
| **External** | temperature, competitor_price |
| **Marketing** | ad_spend, email_campaigns |

### Data Quality Checklist

Before uploading, verify:

- [ ] No blank rows in the middle of data
- [ ] Consistent date frequency (all weekly, all monthly)
- [ ] No text in numeric columns
- [ ] At least 12 data points (ideally 52+ for weekly)
- [ ] Recent data (within last year for best results)

---

## Tips for Better Forecasts

### Data Preparation

| Tip | Why It Helps |
|-----|--------------|
| **More history = better** | 2+ years ideal for seasonality |
| **Clean outliers** | Extreme values distort forecasts |
| **Consistent frequency** | Mix of weekly/monthly confuses models |
| **Include holidays** | Add columns for major shopping events |

### Model Selection

| Your Data | Recommended Models |
|-----------|--------------------|
| Simple trend | Prophet, ARIMA |
| Strong seasonality | Prophet, ETS |
| External factors matter | Prophet + SARIMAX |
| Complex patterns | XGBoost |
| Not sure | Start with Prophet |

### Horizon Selection

| Rule of Thumb | Details |
|---------------|---------|
| **Shorter = more accurate** | 4-week forecast better than 12-week |
| **Match your planning cycle** | If you reforecast monthly, use monthly horizon |
| **Don't exceed history** | 2 years data? Max 6 month forecast |

### Holiday Handling

The system automatically detects and adjusts for:

- Thanksgiving (and weeks before/after)
- Christmas (and weeks before/after)
- Black Friday
- New Year's
- Major US holidays

**Tip:** For non-US holidays, add a column with 1/0 flags.

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "No data to display" | Check CSV has correct column headers |
| Dates not recognized | Use YYYY-MM-DD format |
| Training takes too long | Reduce data size or select fewer models |
| Poor accuracy | Add more history, check for outliers |
| "Column not found" | Verify column names match exactly |

### Error Messages

| Error | Meaning | Fix |
|-------|---------|-----|
| "Insufficient data" | Need 12+ data points | Add more history |
| "Invalid frequency" | Mixed weekly/monthly | Standardize dates |
| "Target contains NaN" | Missing values in target | Fill or remove gaps |
| "Date parsing failed" | Unrecognized date format | Use YYYY-MM-DD |

### Getting Help

1. Check the error message carefully
2. Verify data format matches requirements
3. Try with a smaller dataset first
4. Contact support with:
   - Error message screenshot
   - Sample of your data (remove sensitive info)
   - Steps you followed

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + U` | Upload file |
| `Ctrl/Cmd + Enter` | Start training |
| `Ctrl/Cmd + D` | Download results |
| `Esc` | Close modal/dialog |

---

## Glossary

| Term | Definition |
|------|------------|
| **MAPE** | Mean Absolute Percentage Error - average % difference between forecast and actual |
| **RMSE** | Root Mean Square Error - average absolute difference |
| **R²** | R-squared - how well the model explains variance (0-1) |
| **Horizon** | Number of future periods to forecast |
| **Covariate** | Additional variable that influences the target |
| **Seasonality** | Repeating patterns (weekly, monthly, yearly) |
| **Trend** | Long-term direction (up, down, flat) |
| **Confidence Interval** | Range where true value likely falls |

---

**Version:** 1.3.1
**Last Updated:** January 2026



export interface DataRow {
  [key: string]: string | number;
}

export enum AppStep {
  UPLOAD = 'UPLOAD',
  ANALYSIS = 'ANALYSIS',
  CONFIG = 'CONFIG',
  TRAINING = 'TRAINING',
  RESULTS = 'RESULTS'
}

export interface DatasetAnalysis {
  summary: string;
  suggestedTimeColumn?: string;
  suggestedTargetColumn?: string;
  suggestedGroupColumns?: string[];
  suggestedCovariates?: string[];
  seasonality?: string;
}

export interface ForecastMetrics {
  mape: string;
  rmse: string;
  r2: string;
  cv_mape?: string;     // Cross-validation MAPE (more robust estimate)
  cv_mape_std?: string; // CV MAPE standard deviation
}

export interface CovariateImpact {
  name: string;
  score: number; // Relative importance score (0-100)
  direction: 'positive' | 'negative'; // Correlation direction
}

export type ModelType = 'prophet' | 'arima' | 'exponential_smoothing' | 'sarimax' | 'xgboost';

export type FutureRegressorMethod = 'mean' | 'last_value' | 'linear_trend';

export interface Hyperparameters {
  [key: string]: string | number;
}

export interface ModelRunResult {
  modelType: ModelType;
  modelName: string;
  isBest: boolean;
  metrics: ForecastMetrics;
  hyperparameters: Hyperparameters; // New: Tuned params
  validation: DataRow[];
  forecast: DataRow[];
  experimentUrl?: string; // MLflow experiment URL
  runUrl?: string; // MLflow run URL
}

export interface TuningLog {
  model: string;
  params: string;
  rmse: number;
  status: 'testing' | 'complete';
}

export interface ForecastResult {
  history: DataRow[];     // The data used for training
  results: ModelRunResult[]; // Results for each trained model
  explanation: string;
  pythonCode: string;
  covariateImpacts?: CovariateImpact[]; // Feature importance analysis
  executiveSummary?: string; // AI-generated executive summary
}

// Finance industry best practice MAPE thresholds
export const MAPE_THRESHOLDS = {
  EXCELLENT: 5,    // ≤5% - Excellent forecast (industry gold standard)
  GOOD: 10,        // ≤10% - Good forecast
  ACCEPTABLE: 15,  // ≤15% - Acceptable forecast
  REVIEW: 25,      // ≤25% - Needs review
  // >25% - Significant deviation, requires attention
};

export interface ActualsComparisonRow {
  date: string;
  predicted: number;
  actual: number;
  error: number;           // actual - predicted
  absoluteError: number;   // |actual - predicted|
  percentageError: number; // ((actual - predicted) / actual) * 100
  mape: number;            // |percentage error|
  status: 'excellent' | 'good' | 'acceptable' | 'review' | 'significant_deviation';
  // Context information to help understand forecast errors
  isWeekend: boolean;
  dayOfWeek: string;       // Mon, Tue, Wed, etc.
  contextFlags: string[];  // Active covariates/promos (e.g., "is_holiday", "is_promo")
}

export interface ActualsComparisonResult {
  rows: ActualsComparisonRow[];
  overallMAPE: number;
  overallRMSE: number;
  overallBias: number;     // Average error (positive = under-forecasting, negative = over-forecasting)
  excellentCount: number;  // Count of rows ≤5% MAPE
  goodCount: number;       // Count of rows 5-10% MAPE
  acceptableCount: number; // Count of rows 10-15% MAPE
  reviewCount: number;     // Count of rows 15-25% MAPE
  deviationCount: number;  // Count of rows >25% MAPE
  modelType: string;
}

// ==========================================
// BATCH TRAINING TYPES
// ==========================================

export interface BatchSegment {
  id: string;                           // Unique segment ID (e.g., "region=US | product=Widget")
  filters: Record<string, string | number>; // Filter values for this segment
  data: DataRow[];                      // Filtered data for this segment
  rowCount: number;                     // Number of rows in this segment
}

export interface BatchTrainingProgress {
  segmentId: string;
  status: 'pending' | 'training' | 'completed' | 'failed';
  progress: number;                     // 0-100
  mape?: string;                        // Best model MAPE when completed
  bestModel?: string;                   // Best model type when completed
  error?: string;                       // Error message if failed
  startTime?: number;
  endTime?: number;
}

export interface BatchTrainingResult {
  segmentId: string;
  filters: Record<string, string | number>;
  status: 'success' | 'error';
  result?: ForecastResult;              // Full forecast result if successful
  bestModel?: string;
  metrics?: {
    mape: string;
    rmse: string;
    r2: string;
    cv_mape?: string;
  };
  runId?: string;
  experimentUrl?: string;               // MLflow experiment URL for tracking
  error?: string;
}

export interface BatchTrainingSummary {
  totalSegments: number;
  successful: number;
  failed: number;
  results: BatchTrainingResult[];
  startTime: number;
  endTime?: number;
  // MAPE statistics
  mapeStats?: {
    min: number;
    max: number;
    mean: number;
    median: number;
  };
  // MLflow tracking reference
  batchId?: string;
  experimentName?: string;
}

// ==========================================
// BATCH COMPARISON TYPES
// ==========================================

export interface BatchComparisonRow {
  segmentId: string;
  filters: Record<string, string | number>;
  forecastMAPE: string;                 // Training MAPE from forecast
  actualMAPE: number;                   // MAPE vs actual values
  actualRMSE: number;
  actualBias: number;
  periodsCompared: number;
  status: 'excellent' | 'good' | 'acceptable' | 'review' | 'significant_deviation';
  forecastRunId?: string;
}

export interface BatchComparisonResult {
  rows: BatchComparisonRow[];
  overallMAPE: number;
  totalSegments: number;
  segmentsByStatus: {
    excellent: number;
    good: number;
    acceptable: number;
    review: number;
    significant_deviation: number;
  };
  comparisonDate: string;
}


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

// Data quality and model recommendation types
export interface DataQuality {
  level: 'excellent' | 'good' | 'fair' | 'poor' | 'insufficient';
  score: number;
  description: string;
}

export interface DataStats {
  observations: number;
  yearsOfData: number;
  dateRange: string;
  frequency: string;
  hasGaps: boolean;
  gapCount: number;
}

export interface PatternInfo {
  trend: {
    type: string;
    strength: number;
  };
  seasonality: {
    type: string;
    strength: number;
  };
  hasOutliers: boolean;
  outlierPercentage: number;
}

export interface ModelRecommendation {
  model: string;
  recommended: boolean;
  confidence: number;
  reason: string;
}

export interface DataAnalysisResult {
  dataQuality: DataQuality;
  dataStats: DataStats;
  patterns: PatternInfo;
  modelRecommendations: ModelRecommendation[];
  recommendedModels: string[];
  excludedModels: string[];
  warnings: string[];
  notes: string[];
  overallRecommendation: string;
  hyperparameterFilters: Record<string, Record<string, any>>;
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

export type ModelType = 'prophet' | 'arima' | 'exponential_smoothing' | 'sarimax' | 'xgboost' | 'statsforecast' | 'chronos' | 'ensemble';

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
  holdoutMape?: number | null; // MAPE on holdout set (used for best model selection)
}

export interface TuningLog {
  model: string;
  params: string;
  rmse: number;
  status: 'testing' | 'complete';
}

export interface AutoOptimizeInfo {
  enabled: boolean;
  forecastability_score: number | null;
  grade: string | null;                   // "excellent" | "good" | "fair" | "poor" | "unforecastable"
  training_window_weeks: number | null;
  from_date_applied: string | null;
  log_transform: string | null;           // "always" | "auto" | "never"
  models_selected: string[] | null;
  models_excluded: string[] | null;
  recommended_horizon: number | null;
  max_reliable_horizon: number | null;
  expected_mape_range: number[] | null;   // [low, high]
  growth_pct: number | null;
  summary: string | null;
}

export interface ForecastResult {
  history: DataRow[];     // The data used for training
  results: ModelRunResult[]; // Results for each trained model
  explanation: string;
  pythonCode: string;
  covariateImpacts?: CovariateImpact[]; // Feature importance analysis
  executiveSummary?: string; // AI-generated executive summary
  autoOptimizeInfo?: AutoOptimizeInfo; // Auto-optimization decisions
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

// ==========================================
// DISTRIBUTED TRAINING JOB TYPES
// ==========================================

export type TrainingMode = 'autogluon' | 'statsforecast' | 'neuralforecast' | 'mmf' | 'legacy';

export type JobStatus = 'pending' | 'submitting' | 'running' | 'completed' | 'failed' | 'cancelled' | 'cancelling';

export interface TrainingModeInfo {
  value: TrainingMode;
  name: string;
  description: string;
  speed: 'fast' | 'medium' | 'slow' | 'variable';
  recommended: boolean;
}

export interface JobConfig {
  data: DataRow[];
  time_col: string;
  target_col: string;
  id_col?: string;
  covariates: string[];
  horizon: number;
  frequency: string;
  training_mode: TrainingMode;
  models: string[];
  seasonality_mode: string;
  time_limit: number;
  presets: string;
  season_length?: number;
}

export interface TrainingJob {
  job_id: string;
  status: JobStatus;
  progress: number;
  current_step: string;
  run_id?: string;
  mlflow_run_id?: string;
  created_at?: string;
  submitted_at?: string;
  completed_at?: string;
  results?: JobResults;
  error?: string;
}

export interface JobResults {
  framework: string;
  best_model: string;
  mape: number;
  mlflow_run_id?: string;
  leaderboard?: Array<{
    model: string;
    score_val: number;
    fit_time_marginal: number;
  }>;
  forecast?: DataRow[];
  model_metrics?: Record<string, number>;
}

export interface DelegationStatus {
  enabled: boolean;
  cluster_id?: string;
  message: string;
  training_modes?: TrainingMode[];
}

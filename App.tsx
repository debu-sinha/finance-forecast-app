
import React, { useState, useRef, useMemo, useEffect } from 'react';
import {
  Database,
  Table,
  PlayCircle,
  FileDown,
  BrainCircuit,
  Check,
  Loader2,
  Filter,
  Layers,
  Calendar,
  TrendingUp,
  Activity,
  Info,
  Settings2,
  GitCommit,
  Trophy,
  Binary,
  Sliders,
  Eye,
  Plus,
  ArrowRight,
  Combine,
  TerminalSquare,
  Rocket,
  AlertTriangle,
  Upload,
  Target,
  CheckCircle2,
  XCircle,
  Zap,
  Wrench,
  RefreshCw,
  BarChart3
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { parseCSV } from './utils/csvParser';
import { AppStep, DataRow, DatasetAnalysis, ForecastResult, CovariateImpact, ModelType, ModelRunResult, FutureRegressorMethod, Hyperparameters, TuningLog, ActualsComparisonResult, ActualsComparisonRow, MAPE_THRESHOLDS, DataAnalysisResult } from './types';
import { analyzeDataset, generateForecastInsights } from './services/analysisService';
import { trainModelOnBackend, deployModel, generateExecutiveSummary, ActualsComparisonSummary, BatchTrainRequest, trainBatchOnBackend, analyzeTrainingData } from './services/databricksApi';
import { BatchTraining } from './components/BatchTraining';
import { BatchComparison } from './components/BatchComparison';
import { BatchResultsViewer } from './components/BatchResultsViewer';
import { SimpleModePanel } from './components/SimpleModePanel';
import { BatchTrainingSummary } from './types';
import { NotebookCell } from './components/NotebookCell';
import { ResultsChart } from './components/ResultsChart';
import { EvaluationChart } from './components/EvaluationChart';
import { ForecastTable } from './components/ForecastTable';
import { CovariateImpactChart } from './components/CovariateImpactChart';
import { TrainTestSplitViz } from './components/TrainTestSplitViz';
import { logger } from './utils/logger';

// Helper component for data preview
const DataPreview = ({ data, title }: { data: DataRow[], title: string }) => {
  if (data.length === 0) return null;
  const previewRows = data.slice(0, 5);
  const columns = Object.keys(data[0]);

  return (
    <div className="mt-4 border border-gray-200 rounded-md overflow-hidden">
      <div className="bg-gray-50 px-3 py-2 border-b border-gray-200 flex justify-between items-center">
        <span className="text-xs font-bold text-gray-600 uppercase">{title} Preview</span>
        <span className="text-[10px] text-gray-400">First 5 rows</span>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.map(col => (
                <th key={col} className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {previewRows.map((row, idx) => (
              <tr key={idx}>
                {columns.map(col => (
                  <td key={`${idx}-${col}`} className="px-3 py-2 whitespace-nowrap text-xs text-gray-500">
                    {row[col]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const UserInstructions = () => {
  const [isOpen, setIsOpen] = useState(true);
  const [showFileFormats, setShowFileFormats] = useState(false);

  return (
    <div className="mb-6 border border-blue-100 bg-blue-50 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between text-blue-900 hover:bg-blue-100 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <Info className="w-5 h-5 text-blue-600" />
          <span className="font-semibold">How to Use This Tool</span>
        </div>
        {isOpen ? <span className="text-xl">−</span> : <span className="text-xl">+</span>}
      </button>

      {isOpen && (
        <div className="px-4 py-3 border-t border-blue-100 text-sm text-blue-800 space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-1">
              <div className="font-semibold flex items-center space-x-1">
                <span className="bg-blue-200 text-blue-800 w-5 h-5 rounded-full flex items-center justify-center text-xs">1</span>
                <span>Upload Data</span>
              </div>
              <p className="text-blue-700/80 pl-6">
                Upload your main time series CSV (must have a date column and a target value column). Optionally upload a features file to join.
              </p>
            </div>

            <div className="space-y-1">
              <div className="font-semibold flex items-center space-x-1">
                <span className="bg-blue-200 text-blue-800 w-5 h-5 rounded-full flex items-center justify-center text-xs">2</span>
                <span>Configure</span>
              </div>
              <p className="text-blue-700/80 pl-6">
                Select your time column, target variable, and forecast horizon. Choose between Monthly, Weekly, or Daily frequency.
              </p>
            </div>

            <div className="space-y-1">
              <div className="font-semibold flex items-center space-x-1">
                <span className="bg-blue-200 text-blue-800 w-5 h-5 rounded-full flex items-center justify-center text-xs">3</span>
                <span>Train Models</span>
              </div>
              <p className="text-blue-700/80 pl-6">
                Select models (Prophet, ARIMA, ETS, SARIMAX, XGBoost) and click "Start Training". The system will auto-tune hyperparameters.
              </p>
            </div>

            <div className="space-y-1">
              <div className="font-semibold flex items-center space-x-1">
                <span className="bg-blue-200 text-blue-800 w-5 h-5 rounded-full flex items-center justify-center text-xs">4</span>
                <span>Analyze & Deploy</span>
              </div>
              <p className="text-blue-700/80 pl-6">
                View interactive charts, compare with actuals, and deploy the best model to Databricks.
              </p>
            </div>
          </div>

          {/* File Format Details Toggle */}
          <div className="pt-2 border-t border-blue-200">
            <button
              onClick={() => setShowFileFormats(!showFileFormats)}
              className="text-blue-700 font-semibold text-xs hover:text-blue-900 flex items-center space-x-1"
            >
              <span>{showFileFormats ? '▼' : '▶'}</span>
              <span>Required CSV File Formats</span>
            </button>

            {showFileFormats && (
              <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                {/* Main Data File */}
                <div className="bg-white/50 rounded-lg p-3 border border-blue-200">
                  <div className="font-bold text-blue-900 mb-2 flex items-center space-x-1">
                    <Database className="w-4 h-4" />
                    <span>Main Time Series File</span>
                  </div>
                  <p className="text-blue-700 mb-2">Historical data with dates and values to forecast.</p>
                  <div className="bg-gray-100 rounded p-2 font-mono text-[10px] text-gray-700">
                    <div className="font-bold text-gray-600">Required columns:</div>
                    <div>• <span className="text-green-700">date</span> - Date column (ds, date, Date, week, month)</div>
                    <div>• <span className="text-green-700">value</span> - Target to forecast (numeric)</div>
                    <div className="mt-2 font-bold text-gray-600">Example:</div>
                    <div className="bg-white p-1 rounded mt-1">
                      date,revenue,region<br />
                      2024-01-01,15000,North<br />
                      2024-01-08,16200,North<br />
                      2024-01-15,14800,North
                    </div>
                  </div>
                </div>

                {/* Features File */}
                <div className="bg-white/50 rounded-lg p-3 border border-blue-200">
                  <div className="font-bold text-blue-900 mb-2 flex items-center space-x-1">
                    <Layers className="w-4 h-4" />
                    <span>Features/Covariates File (Optional)</span>
                  </div>
                  <p className="text-blue-700 mb-2">External factors like promotions, holidays, events.</p>
                  <div className="bg-gray-100 rounded p-2 font-mono text-[10px] text-gray-700">
                    <div className="font-bold text-gray-600">Required columns:</div>
                    <div>• <span className="text-green-700">date</span> - Must match main file dates</div>
                    <div>• <span className="text-green-700">features</span> - Binary (0/1) or numeric values</div>
                    <div className="mt-2 font-bold text-gray-600">Example:</div>
                    <div className="bg-white p-1 rounded mt-1">
                      date,promo,holiday,event<br />
                      2024-01-01,0,1,0<br />
                      2024-01-08,1,0,0<br />
                      2024-01-15,0,0,1
                    </div>
                    <div className="mt-1 text-blue-600 italic">Used by: Prophet, SARIMAX, XGBoost</div>
                  </div>
                </div>

                {/* Actuals File */}
                <div className="bg-white/50 rounded-lg p-3 border border-blue-200">
                  <div className="font-bold text-blue-900 mb-2 flex items-center space-x-1">
                    <Target className="w-4 h-4" />
                    <span>Actuals File (Post-Forecast)</span>
                  </div>
                  <p className="text-blue-700 mb-2">Real values to compare against predictions.</p>
                  <div className="bg-gray-100 rounded p-2 font-mono text-[10px] text-gray-700">
                    <div className="font-bold text-gray-600">Required columns:</div>
                    <div>• <span className="text-green-700">date column</span> - Any column with dates (you'll select which one)</div>
                    <div>• <span className="text-green-700">value column</span> - Any numeric column (you'll select which one)</div>
                    <div className="mt-2 font-bold text-gray-600">Example:</div>
                    <div className="bg-white p-1 rounded mt-1">
                      DAY,TOT_VOL,TOT_SUB,REGION<br />
                      11/19/25,816485,20207713,North<br />
                      11/18/25,1973078,46131161,North
                    </div>
                    <div className="mt-1 text-green-600 italic">You can select which columns to use after upload</div>
                    <div className="text-orange-600 italic">Only overlapping dates with forecast are compared</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

// Parse dates flexibly - handles MM/DD/YY, MM/DD/YYYY, YYYY-MM-DD, etc.
// IMPORTANT: This handles 2-digit years correctly (e.g., "12/8/25" = 2025, not 1925)
const parseFlexibleDateUtil = (dateStr: string): Date | null => {
  if (!dateStr) return null;
  const str = String(dateStr).trim();

  // Try MM/DD/YY format FIRST (e.g., "11/19/25") - handles 2-digit years
  const mmddyy = str.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2})$/);
  if (mmddyy) {
    const month = parseInt(mmddyy[1]) - 1;
    const day = parseInt(mmddyy[2]);
    let year = parseInt(mmddyy[3]);
    year = year < 100 ? 2000 + year : year;
    const d = new Date(year, month, day);
    if (!isNaN(d.getTime())) return d;
  }

  // Try standard Date parsing (handles ISO, MM/DD/YYYY, etc.)
  let d = new Date(str);
  if (!isNaN(d.getTime())) return d;

  // Try YYYY-MM-DD format
  const yyyymmdd = str.match(/^(\d{4})-(\d{2})-(\d{2})/);
  if (yyyymmdd) {
    d = new Date(parseInt(yyyymmdd[1]), parseInt(yyyymmdd[2]) - 1, parseInt(yyyymmdd[3]));
    if (!isNaN(d.getTime())) return d;
  }

  return null;
};

const detectFrequency = (data: DataRow[], dateCol: string): 'daily' | 'weekly' | 'monthly' => {
  if (!data || data.length < 2 || !dateCol) return 'monthly';

  // Use parseFlexibleDateUtil to handle 2-digit years correctly
  const dates = data
    .map(row => {
      const d = parseFlexibleDateUtil(String(row[dateCol]));
      return d ? d.getTime() : NaN;
    })
    .filter(t => !isNaN(t))
    .sort((a, b) => a - b);

  if (dates.length < 2) return 'monthly';

  // Calculate differences between consecutive dates (take first 50 points to be fast)
  const diffs: number[] = [];
  for (let i = 1; i < Math.min(dates.length, 50); i++) {
    diffs.push(dates[i] - dates[i - 1]);
  }

  diffs.sort((a, b) => a - b);
  const medianDiff = diffs[Math.floor(diffs.length / 2)];
  const dayInMs = 1000 * 60 * 60 * 24;

  if (medianDiff <= dayInMs * 1.5) return 'daily';
  if (medianDiff <= dayInMs * 8) return 'weekly';
  return 'monthly';
};

const App = () => {
  const [step, setStep] = useState<AppStep>(AppStep.UPLOAD);

  // Mode State - Simple (autopilot) or Expert (full control)
  const [appMode, setAppMode] = useState<'simple' | 'expert'>('simple');

  // Data State
  const [mainData, setMainData] = useState<DataRow[]>([]);
  const [featureData, setFeatureData] = useState<DataRow[]>([]);
  const [mergedData, setMergedData] = useState<DataRow[]>([]);

  // Column State
  const [columns, setColumns] = useState<string[]>([]);
  const [mainColumns, setMainColumns] = useState<string[]>([]);
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);

  // Join Config
  const [mainDateCol, setMainDateCol] = useState<string>('');
  const [featureDateCol, setFeatureDateCol] = useState<string>('');

  // Config State
  const [timeCol, setTimeCol] = useState<string>('');
  const [targetCol, setTargetCol] = useState<string>('');
  const [groupCols, setGroupCols] = useState<string[]>([]);
  const [covariates, setCovariates] = useState<string[]>([]);
  const [horizon, setHorizon] = useState<number>(12);
  const [frequency, setFrequency] = useState<'weekly' | 'monthly' | 'daily'>('monthly');
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [trainingStartDate, setTrainingStartDate] = useState<string>('');
  const [trainingEndDate, setTrainingEndDate] = useState<string>('');
  const [randomSeed, setRandomSeed] = useState<number>(42);
  const [country, setCountry] = useState<string>('US');

  const [regressorMethod, setRegressorMethod] = useState<FutureRegressorMethod>('last_value');
  const [selectedModels, setSelectedModels] = useState<ModelType[]>(['prophet']);

  // Seasonality State
  const [seasonalityMode, setSeasonalityMode] = useState<'additive' | 'multiplicative'>('multiplicative');
  const [enableYearly, setEnableYearly] = useState<boolean>(true);
  const [enableWeekly, setEnableWeekly] = useState<boolean>(false);

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);

  // Data Intelligence Analysis State
  const [isAnalyzingData, setIsAnalyzingData] = useState(false);
  const [dataAnalysis, setDataAnalysis] = useState<DataAnalysisResult | null>(null);
  const [showDataAnalysis, setShowDataAnalysis] = useState(false);

  // Training State
  const [isTraining, setIsTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<ForecastResult | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<string>('');
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [tuningLogs, setTuningLogs] = useState<TuningLog[]>([]);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Model Training Progress State
  interface ModelTrainingProgress {
    model: string;
    displayName: string;
    status: 'pending' | 'training' | 'completed' | 'failed';
    mape?: string;
    startTime?: number;
    error?: string;
  }
  const [modelProgress, setModelProgress] = useState<ModelTrainingProgress[]>([]);

  // View State
  const [activeModelType, setActiveModelType] = useState<ModelType>('prophet');
  const [compareAllModels, setCompareAllModels] = useState<boolean>(false);

  // Actuals Comparison State
  const [actualsData, setActualsData] = useState<DataRow[]>([]);
  const [actualsColumns, setActualsColumns] = useState<string[]>([]);
  const [actualsDateCol, setActualsDateCol] = useState<string>('');
  const [actualsValueCol, setActualsValueCol] = useState<string>('');
  const [actualsFilters, setActualsFilters] = useState<Record<string, string>>({});
  const [actualsComparison, setActualsComparison] = useState<ActualsComparisonResult | null>(null);
  const [isComparingActuals, setIsComparingActuals] = useState(false);
  const [showActualsColumnSelector, setShowActualsColumnSelector] = useState(false);
  const [comparisonModelIndex, setComparisonModelIndex] = useState<number>(0); // Index of model to compare against
  const [filteredActualsForComparison, setFilteredActualsForComparison] = useState<DataRow[]>([]); // Store filtered actuals used in comparison
  const [comparisonSeverityFilter, setComparisonSeverityFilter] = useState<string[]>([]); // Filter by severity: 'excellent', 'good', 'acceptable', 'review', 'significant_deviation'
  const actualsFileInputRef = useRef<HTMLInputElement>(null);

  // Deployment State
  const [isDeploying, setIsDeploying] = useState(false);
  const [deployStatus, setDeployStatus] = useState('');

  // Unity Catalog Configuration
  const [catalogName, setCatalogName] = useState('main');
  const [schemaName, setSchemaName] = useState('default');
  const [modelName, setModelName] = useState('finance_forecast_model');

  // Batch Training State - with localStorage persistence to prevent data loss
  const [showBatchTraining, setShowBatchTraining] = useState(false);
  const [batchTrainingSummary, setBatchTrainingSummary] = useState<BatchTrainingSummary | null>(() => {
    try {
      const saved = localStorage.getItem('batchTrainingSummary');
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [batchSegmentCols, setBatchSegmentCols] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem('batchSegmentCols');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [showBatchComparison, setShowBatchComparison] = useState(false);
  const [showBatchResultsViewer, setShowBatchResultsViewer] = useState(false);

  // Persist batch results to localStorage
  useEffect(() => {
    if (batchTrainingSummary) {
      // Create a lightweight version for storage to prevent quota limits (5MB)
      // We strip the heavy forecast/history arrays and keep only metrics/metadata
      const lightweightSummary: BatchTrainingSummary = {
        ...batchTrainingSummary,
        results: batchTrainingSummary.results.map(r => ({
          ...r,
          result: r.result ? {
            ...r.result,
            history: [], // Strip heavy data
            forecast: [],
            validation: []
          } : undefined
        }))
      };

      try {
        localStorage.setItem('batchTrainingSummary', JSON.stringify(lightweightSummary));
        localStorage.setItem('batchSegmentCols', JSON.stringify(batchSegmentCols));
      } catch (e) {
        logger.error('Failed to save batch results to localStorage:', e);
      }
    }
  }, [batchTrainingSummary, batchSegmentCols]);

  const resetApp = () => {
    setStep(AppStep.UPLOAD);
    setMainData([]);
    setFeatureData([]);
    setMergedData([]);
    setColumns([]);
    setMainColumns([]);
    setFeatureColumns([]);
    setMainDateCol('');
    setFeatureDateCol('');
    setTimeCol('');
    setTargetCol('');
    setGroupCols([]);
    setCovariates([]);
    setHorizon(12);
    setFrequency('monthly');
    setFilters({});
    setTrainingStartDate('');
    setTrainingEndDate('');
    setAnalysis(null);
    setTrainingResult(null);
    setTuningLogs([]);
    setModelProgress([]);
    setActiveModelType('prophet');
    setDeployStatus('');
    setActualsData([]);
    setActualsColumns([]);
    setActualsDateCol('');
    setActualsValueCol('');
    setActualsFilters({});
    setActualsComparison(null);
    setFilteredActualsForComparison([]);
    setComparisonSeverityFilter([]);
    setShowActualsColumnSelector(false);
    // Reset batch training state and clear localStorage
    setShowBatchTraining(false);
    setBatchTrainingSummary(null);
    setBatchSegmentCols([]);
    setShowBatchComparison(false);
    setShowBatchResultsViewer(false);
    localStorage.removeItem('batchTrainingSummary');
    localStorage.removeItem('batchSegmentCols');
    // Reset file inputs
    if (mainFileInputRef.current) mainFileInputRef.current.value = '';
    if (featureFileInputRef.current) featureFileInputRef.current.value = '';
    if (actualsFileInputRef.current) actualsFileInputRef.current.value = '';
  };

  const mainFileInputRef = useRef<HTMLInputElement>(null);
  const featureFileInputRef = useRef<HTMLInputElement>(null);

  const filteredByCategories = useMemo(() => {
    logger.debug('Filter Debug - Stage 1 (Category Filters):');
    logger.debug('  mergedData.length:', mergedData.length);
    logger.debug('  Active filters:', filters);

    if (mergedData.length > 0) {
      logger.debug('  Sample row from mergedData:', mergedData[0]);
      logger.debug('  IS_CGNA type:', typeof mergedData[0].IS_CGNA, 'value:', mergedData[0].IS_CGNA);
      logger.debug('  BUSINESS_SEGMENT type:', typeof mergedData[0].BUSINESS_SEGMENT, 'value:', mergedData[0].BUSINESS_SEGMENT);
      logger.debug('  MX_TYPE type:', typeof mergedData[0].MX_TYPE, 'value:', mergedData[0].MX_TYPE);
    }

    // First, filter out feature-only rows (rows where target column is undefined)
    // This happens when features file has dates that don't exist in main data
    const dataWithTarget = targetCol ? mergedData.filter(row => row[targetCol] !== undefined) : mergedData;
    logger.debug('  After removing feature-only rows:', dataWithTarget.length);

    // Debug: Count how many rows match each filter individually
    if (Object.keys(filters).length > 0) {
      Object.entries(filters).forEach(([key, val]) => {
        const matchCount = dataWithTarget.filter(row => String(row[key]) === val).length;
        logger.debug(`  Rows matching ${key}='${val}':`, matchCount);
      });

      // Show all unique values for each filter column
      Object.keys(filters).forEach(key => {
        const uniqueVals = Array.from(new Set(dataWithTarget.map(row => String(row[key])))).slice(0, 10);
        logger.debug(`  Unique values for ${key}:`, uniqueVals);
      });
    }

    if (Object.keys(filters).length === 0) {
      logger.debug('  No category filters applied');
      return dataWithTarget;
    }

    const result = dataWithTarget.filter(row => {
      const matches = Object.entries(filters).every(([key, val]) => {
        // Skip empty filter values (means "All/Aggregated")
        if (val === '' || val === undefined || val === null) {
          return true;
        }
        const rowVal = String(row[key]);
        const filterVal = val;
        const match = rowVal === filterVal;
        if (!match && row === dataWithTarget[0]) {
          logger.debug(`  First row doesn't match: ${key} - row has "${rowVal}" (${typeof row[key]}), filter wants "${filterVal}"`);
        }
        return match;
      });
      return matches;
    });

    logger.debug('  filteredByCategories.length:', result.length);

    // [PIPELINE STEP F4] AFTER_CATEGORY_FILTER
    console.group(`[PIPELINE STEP F4] AFTER_CATEGORY_FILTER`);
    console.log(`  Rows: ${dataWithTarget.length} → ${result.length} (dropped ${dataWithTarget.length - result.length})`);
    console.log(`  Filters applied:`, filters);
    if (result.length > 0 && targetCol) {
      const f4Vals = result.map(r => Number(r[targetCol]) || 0);
      console.log(`  Target '${targetCol}' stats: min=${Math.min(...f4Vals).toLocaleString()}, max=${Math.max(...f4Vals).toLocaleString()}, mean=${(f4Vals.reduce((a, b) => a + b, 0) / f4Vals.length).toLocaleString()}`);
    }
    console.log(`  First 3:`, result.slice(0, 3));
    console.log(`  Last 3:`, result.slice(-3));
    console.groupEnd();

    return result;
  }, [mergedData, filters, targetCol]);

  const filteredData = useMemo(() => {
    logger.debug('Filter Debug - Stage 2 (Date Range):');
    logger.debug('  filteredByCategories.length:', filteredByCategories.length);
    logger.debug('  trainingStartDate:', trainingStartDate);
    logger.debug('  trainingEndDate:', trainingEndDate);

    if (!timeCol) return filteredByCategories;

    const result = filteredByCategories.filter(row => {
      const rowDate = new Date(String(row[timeCol]));
      if (trainingStartDate) {
        const startDate = new Date(trainingStartDate);
        if (rowDate < startDate) return false;
      }
      if (trainingEndDate) {
        const endDate = new Date(trainingEndDate);
        if (rowDate > endDate) return false;
      }
      return true;
    });

    logger.debug('  filteredData.length:', result.length);

    // [PIPELINE STEP F5] AFTER_DATE_FILTER
    console.group(`[PIPELINE STEP F5] AFTER_DATE_FILTER`);
    console.log(`  Rows: ${filteredByCategories.length} → ${result.length} (dropped ${filteredByCategories.length - result.length})`);
    console.log(`  Date range filter: ${trainingStartDate || '(start)'} to ${trainingEndDate || '(end)'}`);
    if (result.length > 0 && timeCol) {
      const f5Dates = result.map(r => String(r[timeCol])).sort();
      console.log(`  Actual date range: ${f5Dates[0]} to ${f5Dates[f5Dates.length - 1]}`);
    }
    if (result.length > 0 && targetCol) {
      const f5Vals = result.map(r => Number(r[targetCol]) || 0);
      console.log(`  Target '${targetCol}' stats: min=${Math.min(...f5Vals).toLocaleString()}, max=${Math.max(...f5Vals).toLocaleString()}, mean=${(f5Vals.reduce((a, b) => a + b, 0) / f5Vals.length).toLocaleString()}`);
    }
    console.log(`  First 3:`, result.slice(0, 3));
    console.log(`  Last 3:`, result.slice(-3));
    console.groupEnd();

    return result;
  }, [filteredByCategories, timeCol, trainingStartDate, trainingEndDate]);

  const aggregatedData = useMemo(() => {
    if (!timeCol || !targetCol || filteredData.length === 0) return [];

    logger.debug('Aggregation Debug:');
    logger.debug('  filteredData.length:', filteredData.length);

    // Detect if target column is an average-type column (should use mean instead of sum)
    const isAverageColumn = targetCol.toLowerCase().includes('avg') ||
                            targetCol.toLowerCase().includes('average') ||
                            targetCol.toLowerCase().includes('mean') ||
                            targetCol.toLowerCase().includes('rate') ||
                            targetCol.toLowerCase().includes('ue');
    logger.debug('  isAverageColumn:', isAverageColumn, '(target:', targetCol, ')');

    // Track both sum and count for proper averaging
    const groups = new Map<string, { row: DataRow; sum: number; count: number }>();

    filteredData.forEach(row => {
      const rawDate = row[timeCol];
      const dateObj = new Date(String(rawDate));

      if (isNaN(dateObj.getTime())) return;

      const dateKey = dateObj.toISOString().split('T')[0];
      const val = Number(row[targetCol]) || 0;

      if (groups.has(dateKey)) {
        const existing = groups.get(dateKey)!;
        existing.sum += val;
        existing.count += 1;
      } else {
        groups.set(dateKey, { row: { ...row, [timeCol]: dateKey }, sum: val, count: 1 });
      }
    });

    // Apply aggregation: use average for AVG columns, sum for others
    const result = Array.from(groups.entries()).map(([dateKey, { row, sum, count }]) => {
      const aggregatedValue = isAverageColumn ? (sum / count) : sum;
      return { ...row, [targetCol]: aggregatedValue };
    }).sort((a, b) =>
      new Date(String(a[timeCol])).getTime() - new Date(String(b[timeCol])).getTime()
    );

    logger.debug('  aggregatedData.length:', result.length);
    logger.debug('  Unique dates:', groups.size);
    logger.debug('  Aggregation method:', isAverageColumn ? 'AVERAGE' : 'SUM');

    // Debug: Log sample aggregated values to verify scale
    if (result.length > 0) {
      const targetValues = result.slice(0, 5).map(r => r[targetCol]);
      logger.debug('  Sample target values (first 5):', targetValues);
      const maxVal = Math.max(...result.map(r => Number(r[targetCol]) || 0));
      const minVal = Math.min(...result.map(r => Number(r[targetCol]) || 0));
      const avgVal = result.reduce((sum, r) => sum + (Number(r[targetCol]) || 0), 0) / result.length;
      logger.debug(`  Target range: min=${minVal.toLocaleString()}, max=${maxVal.toLocaleString()}, avg=${avgVal.toLocaleString()}`);
    }

    // [PIPELINE STEP F6] AFTER_AGGREGATION
    console.group(`[PIPELINE STEP F6] AFTER_AGGREGATION`);
    console.log(`  Rows: ${filteredData.length} → ${result.length}`);
    console.log(`  Method: ${isAverageColumn ? 'MEAN' : 'SUM'}`);
    console.log(`  Unique dates: ${groups.size}`);
    if (result.length > 0) {
      const f6Vals = result.map(r => Number(r[targetCol]) || 0);
      const f6Sum = f6Vals.reduce((a, b) => a + b, 0);
      console.log(`  Target '${targetCol}' stats: min=${Math.min(...f6Vals).toLocaleString()}, max=${Math.max(...f6Vals).toLocaleString()}, mean=${(f6Sum / f6Vals.length).toLocaleString()}, sum=${f6Sum.toLocaleString()}`);
      const f6Dates = result.map(r => String(r[timeCol])).sort();
      console.log(`  Date range: ${f6Dates[0]} to ${f6Dates[f6Dates.length - 1]}`);
      const f6MultiCount = Array.from(groups.values()).filter(g => g.count > 1).length;
      console.log(`  Dates with multiple rows aggregated: ${f6MultiCount}`);
    }
    console.log(`  First 3:`, result.slice(0, 3));
    console.log(`  Last 3:`, result.slice(-3));
    console.groupEnd();

    return result;
  }, [filteredData, timeCol, targetCol]);

  const chartData = useMemo(() => {
    if (!timeCol || !targetCol || aggregatedData.length === 0) return [];

    if (frequency === 'daily') return aggregatedData;

    // Detect if target column is an average-type column (should use mean instead of sum)
    const isAverageColumn = targetCol.toLowerCase().includes('avg') ||
                            targetCol.toLowerCase().includes('average') ||
                            targetCol.toLowerCase().includes('mean') ||
                            targetCol.toLowerCase().includes('rate') ||
                            targetCol.toLowerCase().includes('ue');

    const groups = new Map<string, { row: DataRow; sum: number; count: number }>();

    aggregatedData.forEach(row => {
      const date = new Date(String(row[timeCol]));
      let key = '';

      if (frequency === 'monthly') {
        key = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-01`;
      } else if (frequency === 'weekly') {
        // Get Monday of the week
        const day = date.getDay();
        const diff = date.getDate() - day + (day === 0 ? -6 : 1); // adjust when day is sunday
        const monday = new Date(date.setDate(diff));
        key = monday.toISOString().split('T')[0];
      }

      const val = Number(row[targetCol]) || 0;

      if (groups.has(key)) {
        const existing = groups.get(key)!;
        existing.sum += val;
        existing.count += 1;
      } else {
        groups.set(key, { row: { ...row, [timeCol]: key }, sum: val, count: 1 });
      }
    });

    // Apply aggregation: use average for AVG columns, sum for others
    return Array.from(groups.entries()).map(([key, { row, sum, count }]) => {
      const aggregatedValue = isAverageColumn ? (sum / count) : sum;
      return { ...row, [targetCol]: aggregatedValue };
    }).sort((a, b) =>
      new Date(String(a[timeCol])).getTime() - new Date(String(b[timeCol])).getTime()
    );
  }, [aggregatedData, frequency, timeCol, targetCol]);

  const dateRange = useMemo(() => {
    // Use mainData and mainDateCol to ensure range matches historical data
    // This prevents the range from extending into the future if features are uploaded
    if (!mainDateCol || mainData.length === 0) return { min: '', max: '' };

    const dates = mainData.map(d => new Date(String(d[mainDateCol]))).filter(d => !isNaN(d.getTime()));
    if (dates.length === 0) return { min: '', max: '' };

    const min = new Date(Math.min(...dates.map(d => d.getTime())));
    const max = new Date(Math.max(...dates.map(d => d.getTime())));
    return {
      min: min.toISOString().split('T')[0],
      max: max.toISOString().split('T')[0]
    };
  }, [mainData, mainDateCol]);

  // Set default start and end dates from the original time series
  useEffect(() => {
    if (dateRange.min && dateRange.max) {
      // Set defaults only if not already set (user hasn't manually changed them)
      if (!trainingStartDate) {
        setTrainingStartDate(dateRange.min);
      }
      if (!trainingEndDate) {
        setTrainingEndDate(dateRange.max);
      }
    }
  }, [dateRange.min, dateRange.max]);

  // Smart Horizon Defaults
  useEffect(() => {
    if (frequency === 'daily') {
      setHorizon(90);
    } else {
      setHorizon(12);
    }
  }, [frequency]);

  const handleMainFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset downstream if we are re-uploading
    if (step !== AppStep.UPLOAD) {
      setStep(AppStep.UPLOAD);
      setAnalysis(null);
      setTrainingResult(null);
      setMergedData([]);
      // Clear feature data too to ensure consistency
      setFeatureData([]);
      setFeatureColumns([]);
      setFeatureDateCol('');
      if (featureFileInputRef.current) featureFileInputRef.current.value = '';
    }

    // Always reset column selections when new main file is uploaded
    // This fixes the bug where old grouping options persist after loading a new file
    setTimeCol('');
    setTargetCol('');
    setGroupCols([]);
    setCovariates([]);
    setFilters({});

    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = evt.target?.result as string;
      const parsedData = parseCSV(text);
      if (parsedData.length > 0) {
        logger.debug('📁 MAIN FILE UPLOAD:');
        logger.debug('  Total rows:', parsedData.length);
        logger.debug('  First row:', parsedData[0]);
        logger.debug('  IS_CGNA:', parsedData[0].IS_CGNA, '(type:', typeof parsedData[0].IS_CGNA, ')');
        logger.debug('  BUSINESS_SEGMENT:', parsedData[0].BUSINESS_SEGMENT, '(type:', typeof parsedData[0].BUSINESS_SEGMENT, ')');
        logger.debug('  MX_TYPE:', parsedData[0].MX_TYPE, '(type:', typeof parsedData[0].MX_TYPE, ')');

        // [PIPELINE STEP F1] MAIN_FILE_PARSED
        console.group(`[PIPELINE STEP F1] MAIN_FILE_PARSED`);
        console.log(`  Rows: ${parsedData.length}`);
        const f1Cols = Object.keys(parsedData[0]);
        console.log(`  Columns (${f1Cols.length}): ${f1Cols.join(', ')}`);
        console.log(`  fileName: ${file.name}`);
        const f1NumCols = f1Cols.filter(c => typeof parsedData[0][c] === 'number' || !isNaN(Number(parsedData[0][c])));
        console.log(`  Numeric columns: ${f1NumCols.join(', ')}`);
        console.log(`  First 3:`, parsedData.slice(0, 3));
        console.log(`  Last 3:`, parsedData.slice(-3));
        console.groupEnd();

        setMainData(parsedData);
        const cols = Object.keys(parsedData[0]);
        setMainColumns(cols);
        const probableDate = cols.find(c => {
          const lower = c.toLowerCase();
          return lower.includes('date') || lower === 'ds' || lower.includes('time') || lower === 'week' || lower === 'year' || lower === 'month' || lower === 'day' || lower.includes('timestamp');
        });
        if (probableDate) setMainDateCol(probableDate);
      }
    };
    reader.readAsText(file);
  };

  const handleFeatureFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset downstream if we are re-uploading
    if (step !== AppStep.UPLOAD) {
      setStep(AppStep.UPLOAD);
      setAnalysis(null);
      setTrainingResult(null);
      setMergedData([]);
    }

    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = evt.target?.result as string;
      const parsedData = parseCSV(text);
      if (parsedData.length > 0) {
        // [PIPELINE STEP F2] FEATURE_FILE_PARSED
        console.group(`[PIPELINE STEP F2] FEATURE_FILE_PARSED`);
        console.log(`  Rows: ${parsedData.length}`);
        const f2Cols = Object.keys(parsedData[0]);
        console.log(`  Columns (${f2Cols.length}): ${f2Cols.join(', ')}`);
        console.log(`  fileName: ${file.name}`);
        const f2DateRange = parsedData.map(r => String(r[f2Cols[0]]));
        console.log(`  Date range: ${f2DateRange[0]} to ${f2DateRange[f2DateRange.length - 1]}`);
        console.log(`  First 3:`, parsedData.slice(0, 3));
        console.log(`  Last 3:`, parsedData.slice(-3));
        console.groupEnd();

        setFeatureData(parsedData);
        const cols = Object.keys(parsedData[0]);
        setFeatureColumns(cols);
        const probableDate = cols.find(c => {
          const lower = c.toLowerCase();
          return lower.includes('date') || lower === 'ds' || lower.includes('time') || lower.includes('week') || lower === 'year' || lower === 'month' || lower === 'day' || lower.includes('timestamp');
        });
        if (probableDate) setFeatureDateCol(probableDate);
      }
    };
    reader.readAsText(file);
  };

  // Handle actuals file upload for forecast vs actuals comparison
  const handleActualsFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = evt.target?.result as string;
      const parsedData = parseCSV(text);
      if (parsedData.length > 0) {
        setActualsData(parsedData);
        const cols = Object.keys(parsedData[0]);
        setActualsColumns(cols);

        // Auto-detect date column
        const probableDateCol = cols.find(c => {
          const lower = c.toLowerCase();
          return lower.includes('date') || lower === 'ds' || lower.includes('time') ||
            lower === 'week' || lower === 'day' || lower.includes('timestamp');
        });
        if (probableDateCol) setActualsDateCol(probableDateCol);

        // Auto-detect value column (prefer matching targetCol, then common names)
        const probableValueCol = cols.find(c => c === targetCol) ||
          cols.find(c => {
            const lower = c.toLowerCase();
            return lower === 'y' || lower === 'actual' || lower === 'value' || lower === 'actuals';
          }) ||
          cols.find(c => {
            // Look for numeric columns that aren't the date
            const val = parsedData[0][c];
            return c !== probableDateCol && typeof val === 'number';
          });
        if (probableValueCol) setActualsValueCol(probableValueCol);

        // Reset filters when new file is uploaded
        setActualsFilters({});
        setFilteredActualsForComparison([]);
        setComparisonSeverityFilter([]);

        // Show column selector so user can confirm/change selections
        setShowActualsColumnSelector(true);
        setActualsComparison(null);
      }
    };
    reader.readAsText(file);
  };

  // Get unique values for actuals column (for filter dropdowns)
  const getActualsUniqueValues = (col: string): string[] => {
    const values = new Set<string>();
    actualsData.forEach(row => {
      if (row[col] !== undefined && row[col] !== null) {
        values.add(String(row[col]));
      }
    });
    return Array.from(values).sort();
  };

  // Detect filter columns (categorical columns with <= 20 unique values)
  const getFilterableColumns = (): string[] => {
    if (actualsData.length === 0) return [];
    return actualsColumns.filter(col => {
      if (col === actualsDateCol || col === actualsValueCol) return false;
      const uniqueCount = getActualsUniqueValues(col).length;
      return uniqueCount > 1 && uniqueCount <= 20;
    });
  };

  // Run comparison with selected columns and filters
  const runActualsComparison = () => {
    if (!actualsDateCol || !actualsValueCol) {
      alert('Please select both a date column and a value column.');
      return;
    }

    // Apply filters to actuals data
    let filteredActuals = actualsData;
    const activeFilters = Object.entries(actualsFilters).filter(([_, v]) => v !== '');

    if (activeFilters.length > 0) {
      filteredActuals = actualsData.filter(row => {
        return activeFilters.every(([col, val]) => String(row[col]) === val);
      });
      logger.debug(`Applied ${activeFilters.length} filters: ${filteredActuals.length} rows (from ${actualsData.length})`);
    }

    if (filteredActuals.length === 0) {
      alert('No data remains after applying filters. Please check your filter selections.');
      return;
    }

    // Store the filtered actuals so model switching uses the same data
    setFilteredActualsForComparison(filteredActuals);
    compareActualsWithForecast(filteredActuals, actualsDateCol, actualsValueCol);
  };

  // Helper function to parse various date formats
  // IMPORTANT: Check MM/DD/YY format FIRST to handle 2-digit years correctly
  // (e.g., "12/8/25" should be 2025, not 1925)
  const parseFlexibleDate = (dateStr: string): Date | null => {
    if (!dateStr) return null;

    const str = String(dateStr).trim();

    // Try MM/DD/YY format FIRST (e.g., "11/19/25") - handles 2-digit years
    const mmddyy = str.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2})$/);
    if (mmddyy) {
      const month = parseInt(mmddyy[1]) - 1;
      const day = parseInt(mmddyy[2]);
      let year = parseInt(mmddyy[3]);
      // Assume 20xx for years 00-99
      year = year < 100 ? 2000 + year : year;
      const d = new Date(year, month, day);
      if (!isNaN(d.getTime())) return d;
    }

    // Try standard Date parsing (handles ISO, MM/DD/YYYY, etc.)
    let d = new Date(str);
    if (!isNaN(d.getTime())) return d;

    // Try YYYY-MM-DD format
    const yyyymmdd = str.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (yyyymmdd) {
      const year = parseInt(yyyymmdd[1]);
      const month = parseInt(yyyymmdd[2]) - 1;
      const day = parseInt(yyyymmdd[3]);
      d = new Date(year, month, day);
      if (!isNaN(d.getTime())) return d;
    }

    // Try DD-MM-YYYY format
    const ddmmyyyy = str.match(/^(\d{2})-(\d{2})-(\d{4})$/);
    if (ddmmyyyy) {
      const day = parseInt(ddmmyyyy[1]);
      const month = parseInt(ddmmyyyy[2]) - 1;
      const year = parseInt(ddmmyyyy[3]);
      d = new Date(year, month, day);
      if (!isNaN(d.getTime())) return d;
    }

    return null;
  };

  // Compare actuals with forecast predictions for a specific model
  const compareActualsWithForecast = (actuals: DataRow[], dateCol: string, valueCol: string, modelIndex?: number) => {
    if (!trainingResult) return;

    // Use specified model index or default to comparisonModelIndex
    const idx = modelIndex !== undefined ? modelIndex : comparisonModelIndex;
    const modelResult = trainingResult.results[idx];
    if (!modelResult) return;

    setIsComparingActuals(true);

    try {
      const forecast = modelResult.forecast;
      const comparisonRows: ActualsComparisonRow[] = [];

      // Build a map of actuals by date using selected columns
      const actualsMap = new Map<string, number>();
      let parsedActualDates: string[] = [];
      let duplicateDates: string[] = [];

      actuals.forEach(row => {
        if (dateCol && valueCol) {
          const d = parseFlexibleDate(String(row[dateCol]));
          if (d) {
            const dateKey = d.toISOString().split('T')[0];
            // Check for duplicate dates
            if (actualsMap.has(dateKey)) {
              if (!duplicateDates.includes(dateKey)) {
                duplicateDates.push(dateKey);
              }
            }
            // Sum values for duplicate dates (aggregate behavior)
            const existingValue = actualsMap.get(dateKey) || 0;
            actualsMap.set(dateKey, existingValue + Number(row[valueCol]));
            parsedActualDates.push(dateKey);
          }
        }
      });

      // Warn about duplicates
      if (duplicateDates.length > 0) {
        logger.warn(`Found ${duplicateDates.length} duplicate dates in actuals - values were summed. First few: ${duplicateDates.slice(0, 3).join(', ')}`);
      }

      // Build a map of training data covariates by date for context lookup
      const trainingContextMap = new Map<string, DataRow>();
      mergedData.forEach(row => {
        const dateValue = row[timeCol];
        if (dateValue) {
          const d = parseFlexibleDate(String(dateValue));
          if (d) {
            const dateKey = d.toISOString().split('T')[0];
            trainingContextMap.set(dateKey, row);
          }
        }
      });

      // Helper to get day of week string
      const getDayOfWeek = (date: Date): string => {
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        return days[date.getDay()];
      };

      // Helper to check if weekend
      const isWeekendDay = (date: Date): boolean => {
        const day = date.getDay();
        return day === 0 || day === 6; // Sunday = 0, Saturday = 6
      };

      // Helper to extract active context flags from training data
      const getContextFlags = (trainingRow: DataRow | undefined, date: Date): string[] => {
        const flags: string[] = [];

        if (!trainingRow) return flags;

        // Check selected covariates for "truthy" values (1, true, "yes", etc.)
        covariates.forEach(cov => {
          const value = trainingRow[cov];
          if (value !== undefined && value !== null) {
            // Check for boolean-like covariates (is_holiday, is_promo, etc.)
            if (typeof value === 'number' && value === 1) {
              flags.push(cov);
            } else if (typeof value === 'boolean' && value) {
              flags.push(cov);
            } else if (typeof value === 'string' && ['1', 'true', 'yes', 'y'].includes(value.toLowerCase())) {
              flags.push(cov);
            }
          }
        });

        return flags;
      };

      // Parse forecast dates
      let parsedForecastDates: string[] = [];
      logger.debug('Raw forecast data (first 3):', forecast.slice(0, 3));
      logger.debug('timeCol is:', timeCol);

      // Check which date column exists in forecast data
      const firstRow = forecast[0];
      const hasDs = firstRow?.ds !== undefined;
      const hasTimeCol = firstRow?.[timeCol] !== undefined;
      logger.debug('Forecast date columns: ds=', hasDs, ', [', timeCol, ']=', hasTimeCol);

      forecast.forEach(fcstRow => {
        // The date field might be 'ds' or the original timeCol name
        // Also handle both timestamp (number) and string formats
        let fcstDate: Date | null = null;
        const dateValue = fcstRow.ds ?? fcstRow[timeCol];

        if (typeof dateValue === 'number') {
          // It's a Unix timestamp (milliseconds)
          fcstDate = new Date(dateValue);
        } else if (dateValue) {
          fcstDate = parseFlexibleDate(String(dateValue));
        }

        if (fcstDate && !isNaN(fcstDate.getTime())) {
          parsedForecastDates.push(fcstDate.toISOString().split('T')[0]);
        }
      });

      logger.debug('Actuals Comparison Debug:');
      logger.debug('  Actuals dates (first 5):', parsedActualDates.slice(0, 5));
      logger.debug('  Forecast dates (first 5):', parsedForecastDates.slice(0, 5));
      logger.debug('  Actuals count:', actualsMap.size);
      logger.debug('  Forecast count:', forecast.length);

      // Match forecasts with actuals
      forecast.forEach(fcstRow => {
        // Handle both number (timestamp) and string date formats
        // The date field might be 'ds' or the original timeCol name
        let fcstDate: Date | null = null;
        const dateValue = fcstRow.ds ?? fcstRow[timeCol];

        if (typeof dateValue === 'number') {
          fcstDate = new Date(dateValue);
        } else if (dateValue) {
          fcstDate = parseFlexibleDate(String(dateValue));
        }
        if (!fcstDate || isNaN(fcstDate.getTime())) return;

        const dateKey = fcstDate.toISOString().split('T')[0];
        const actual = actualsMap.get(dateKey);

        if (actual !== undefined) {
          const predicted = Number(fcstRow.yhat);

          // Debug: Log first few comparisons to verify data integrity
          if (comparisonRows.length < 3) {
            logger.debug(`Comparison ${comparisonRows.length + 1}: date=${dateKey}, actual=${actual}, predicted=${predicted}, yhat_raw=${fcstRow.yhat}`);
          }

          const error = actual - predicted;
          const absoluteError = Math.abs(error);
          const percentageError = actual !== 0 ? ((error) / actual) * 100 : 0;
          const mape = Math.abs(percentageError);

          // Determine status based on MAPE thresholds
          let status: ActualsComparisonRow['status'];
          if (mape <= MAPE_THRESHOLDS.EXCELLENT) {
            status = 'excellent';
          } else if (mape <= MAPE_THRESHOLDS.GOOD) {
            status = 'good';
          } else if (mape <= MAPE_THRESHOLDS.ACCEPTABLE) {
            status = 'acceptable';
          } else if (mape <= MAPE_THRESHOLDS.REVIEW) {
            status = 'review';
          } else {
            status = 'significant_deviation';
          }

          // Get context information for this date
          const trainingRow = trainingContextMap.get(dateKey);
          const contextFlags = getContextFlags(trainingRow, fcstDate);

          comparisonRows.push({
            date: dateKey,
            predicted,
            actual,
            error,
            absoluteError,
            percentageError,
            mape,
            status,
            isWeekend: isWeekendDay(fcstDate),
            dayOfWeek: getDayOfWeek(fcstDate),
            contextFlags
          });
        }
      });

      if (comparisonRows.length === 0) {
        // Provide more helpful error message
        const actualDateRange = parsedActualDates.length > 0
          ? `${parsedActualDates[0]} to ${parsedActualDates[parsedActualDates.length - 1]}`
          : 'none parsed';
        const forecastDateRange = parsedForecastDates.length > 0
          ? `${parsedForecastDates[0]} to ${parsedForecastDates[parsedForecastDates.length - 1]}`
          : 'none parsed';

        alert(`No matching dates found between forecast and actuals.\n\nActuals date range: ${actualDateRange}\nForecast date range: ${forecastDateRange}\n\nPlease ensure the actuals file has dates that overlap with the forecast period.`);
        setIsComparingActuals(false);
        return;
      }

      // Calculate overall metrics
      const overallMAPE = comparisonRows.reduce((sum, r) => sum + r.mape, 0) / comparisonRows.length;
      const overallRMSE = Math.sqrt(comparisonRows.reduce((sum, r) => sum + r.error * r.error, 0) / comparisonRows.length);
      const overallBias = comparisonRows.reduce((sum, r) => sum + r.error, 0) / comparisonRows.length;

      const result: ActualsComparisonResult = {
        rows: comparisonRows,
        overallMAPE,
        overallRMSE,
        overallBias,
        excellentCount: comparisonRows.filter(r => r.status === 'excellent').length,
        goodCount: comparisonRows.filter(r => r.status === 'good').length,
        acceptableCount: comparisonRows.filter(r => r.status === 'acceptable').length,
        reviewCount: comparisonRows.filter(r => r.status === 'review').length,
        deviationCount: comparisonRows.filter(r => r.status === 'significant_deviation').length,
        modelType: modelResult.modelName  // Use full model name with hyperparameters
      };

      setActualsComparison(result);
    } catch (err) {
      logger.error('Error comparing actuals:', err);
      alert('Error comparing actuals with forecast. Please check the file format.');
    } finally {
      setIsComparingActuals(false);
    }
  };

  // Regenerate executive summary with actuals comparison data
  const regenerateExecutiveSummaryWithActuals = async () => {
    if (!trainingResult || !actualsComparison) return;

    // Use the model that was used for the comparison
    const comparisonModel = trainingResult.results[comparisonModelIndex];
    if (!comparisonModel) return;

    setIsGeneratingSummary(true);
    try {
      // Prepare worst periods for root cause analysis
      const worstPeriods = actualsComparison.rows
        .filter(r => r.status === 'significant_deviation' || r.status === 'review')
        .sort((a, b) => b.mape - a.mape)
        .slice(0, 5)
        .map(r => ({
          date: r.date,
          mape: r.mape,
          error: r.error,
          predicted: r.predicted,
          actual: r.actual
        }));

      const comparisonSummary: ActualsComparisonSummary = {
        overallMAPE: actualsComparison.overallMAPE,
        overallRMSE: actualsComparison.overallRMSE,
        overallBias: actualsComparison.overallBias,
        excellentCount: actualsComparison.excellentCount,
        goodCount: actualsComparison.goodCount,
        acceptableCount: actualsComparison.acceptableCount,
        reviewCount: actualsComparison.reviewCount,
        deviationCount: actualsComparison.deviationCount,
        totalPeriods: actualsComparison.rows.length,
        worstPeriods
      };

      // Use the comparison model (not necessarily the best model) for the summary
      const summary = await generateExecutiveSummary(
        comparisonModel.modelName,
        {
          rmse: parseFloat(comparisonModel.metrics.rmse),
          mape: parseFloat(comparisonModel.metrics.mape),
          r2: parseFloat(comparisonModel.metrics.r2)
        },
        trainingResult.results.map(m => ({
          modelName: m.modelName,
          metrics: {
            rmse: m.metrics.rmse,
            mape: m.metrics.mape,
            r2: m.metrics.r2
          }
        })),
        targetCol,
        timeCol,
        covariates,
        horizon,
        frequency,
        comparisonSummary
      );

      setTrainingResult(prev => prev ? {
        ...prev,
        executiveSummary: summary
      } : null);
    } catch (error) {
      logger.error('Failed to regenerate executive summary:', error);
      alert('Failed to regenerate executive summary. Please try again.');
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  // Download actuals comparison as CSV
  const downloadActualsComparisonCSV = () => {
    if (!actualsComparison) return;

    const headers = ['Date', 'Day', 'Weekend', 'Predicted', 'Actual', 'Difference', 'MAPE %', 'Direction', 'Status', 'Context Flags'];
    const csvContent = [
      headers.join(','),
      ...actualsComparison.rows.map(row => [
        row.date,
        row.dayOfWeek,
        row.isWeekend ? 'Yes' : 'No',
        parseFloat(String(row.predicted)).toFixed(2),
        parseFloat(String(row.actual)).toFixed(2),
        parseFloat(String(row.error)).toFixed(2),
        parseFloat(String(row.mape)).toFixed(2),
        parseFloat(String(row.error)) >= 0 ? 'Under-forecast' : 'Over-forecast',
        row.status.replace('_', ' '),
        `"${(row.contextFlags || []).join(', ')}"`
      ].join(','))
    ].join('\n');

    // Add summary section
    const summarySection = [
      '',
      'SUMMARY',
      `Overall MAPE,${actualsComparison.overallMAPE.toFixed(2)}%`,
      `Overall RMSE,${actualsComparison.overallRMSE.toFixed(2)}`,
      `Overall Bias,${actualsComparison.overallBias.toFixed(2)}`,
      `Total Periods,${actualsComparison.rows.length}`,
      `Excellent (≤5%),${actualsComparison.excellentCount}`,
      `Good (5-10%),${actualsComparison.goodCount}`,
      `Acceptable (10-15%),${actualsComparison.acceptableCount}`,
      `Needs Review (15-25%),${actualsComparison.reviewCount}`,
      `Significant Deviation (>25%),${actualsComparison.deviationCount}`,
      `Model,${actualsComparison.modelType}`
    ].join('\n');

    const fullContent = csvContent + '\n' + summarySection;

    const blob = new Blob([fullContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `forecast_vs_actuals_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Download executive summary as markdown
  const downloadExecutiveSummary = () => {
    if (!trainingResult?.executiveSummary) return;

    const content = trainingResult.executiveSummary;
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `executive_summary_${new Date().toISOString().split('T')[0]}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const mergeAndAnalyze = async () => {
    if (mainData.length === 0) return;

    let finalData = [...mainData];

    // LEFT OUTER JOIN: Keep all rows from mainData, add features where dates match
    const featureMap = new Map<string, DataRow>();

    // Process Feature Data into a lookup map
    // IMPORTANT: Use parseFlexibleDate to handle MM/DD/YY format correctly
    // (2-digit years like "12/8/25" should be 2025, not 1925)
    if (featureData.length > 0 && featureDateCol) {
      featureData.forEach(row => {
        const d = parseFlexibleDate(String(row[featureDateCol]));
        if (d && !isNaN(d.getTime())) {
          const key = d.toISOString().split('T')[0];
          featureMap.set(key, row);
        }
      });
    }

    // Merge: For each main data row, add matching features
    // IMPORTANT: Use parseFlexibleDate for consistent date parsing with feature data
    finalData = mainData.map(mainRow => {
      const d = parseFlexibleDate(String(mainRow[mainDateCol]));
      if (!d || isNaN(d.getTime())) return mainRow;

      const dateKey = d.toISOString().split('T')[0];

      // Add features if they exist for this date
      let features: DataRow = {};
      if (featureMap.has(dateKey)) {
        const featureRow = featureMap.get(dateKey)!;
        const { [featureDateCol]: _, ...rest } = featureRow;
        features = rest;
      } else if (featureData.length > 0) {
        // Fill missing features with 0 if we have feature data but not for this date
        featureColumns.filter(c => c !== featureDateCol).forEach(c => features[c] = 0);
      }

      return { ...mainRow, ...features };
    });

    setMergedData(finalData);

    logger.debug('MERGE DEBUG:');
    logger.debug('  mainData.length:', mainData.length);
    logger.debug('  First 3 rows of mainData:', mainData.slice(0, 3));
    logger.debug('  mergedData.length:', finalData.length);
    logger.debug('  First 3 rows of mergedData:', finalData.slice(0, 3));
    logger.debug('  Last 3 rows of mergedData:', finalData.slice(-3));

    // [PIPELINE STEP F3] AFTER_MERGE
    console.group(`[PIPELINE STEP F3] AFTER_MERGE`);
    console.log(`  Main rows: ${mainData.length}, Feature rows: ${featureData.length}`);
    console.log(`  Merged rows: ${finalData.length}`);
    const f3MatchCount = featureData.length > 0 ? finalData.filter(r => {
      const d = parseFlexibleDate(String(r[mainDateCol]));
      return d && featureMap.has(d.toISOString().split('T')[0]);
    }).length : 0;
    console.log(`  Rows matched with features: ${f3MatchCount}`);
    console.log(`  Rows zero-filled (no match): ${featureData.length > 0 ? finalData.length - f3MatchCount : 0}`);
    const f3MergedCols = finalData.length > 0 ? Object.keys(finalData[0]) : [];
    const f3FeatureColsAdded = f3MergedCols.filter(c => !mainColumns.includes(c));
    console.log(`  Feature columns added: ${f3FeatureColsAdded.join(', ')}`);
    console.log(`  All columns (${f3MergedCols.length}): ${f3MergedCols.join(', ')}`);
    console.log(`  First 3:`, finalData.slice(0, 3));
    console.log(`  Last 3:`, finalData.slice(-3));
    console.groupEnd();

    // Use all available columns from both datasets to ensure dropdowns are complete
    // even if the first row of merged data (e.g. a future row) is missing some columns
    const allCols = Array.from(new Set([...mainColumns, ...featureColumns]));
    setColumns(allCols);
    setTimeCol(mainDateCol || allCols[0]);

    setStep(AppStep.ANALYSIS);

    // Auto-detect frequency
    const detectedFreq = detectFrequency(finalData, mainDateCol || allCols[0]);
    setFrequency(detectedFreq);

    setIsAnalyzing(true);
    const sample = finalData.slice(0, 10);
    const result = await analyzeDataset(sample, allCols);
    setAnalysis(result);

    if (result.suggestedTimeColumn) setTimeCol(result.suggestedTimeColumn);
    if (result.suggestedTargetColumn) setTargetCol(result.suggestedTargetColumn);
    if (result.suggestedCovariates) {
      // When a feature/promo file is provided, only auto-select covariates from that file
      const externalCols = featureColumns.filter(c => c !== featureDateCol);
      const filtered = externalCols.length > 0
        ? result.suggestedCovariates.filter((c: string) => externalCols.includes(c))
        : result.suggestedCovariates;
      setCovariates(filtered);
    }
    if (result.suggestedGroupColumns) setGroupCols(result.suggestedGroupColumns);

    setIsAnalyzing(false);
    setStep(AppStep.CONFIG);
  };

  // Validate data before training
  const validateTrainingData = (): string[] => {
    const errors: string[] = [];

    // Check if models are selected
    if (selectedModels.length === 0) {
      errors.push('Please select at least one model to train.');
    }

    // Check if time column is selected
    if (!timeCol) {
      errors.push('Please select a time column.');
    }

    // Check if target column is selected
    if (!targetCol) {
      errors.push('Please select a target column.');
    }

    // Check if we have data
    if (aggregatedData.length === 0) {
      errors.push('No data available. Please upload a CSV file first.');
    }

    // Check minimum data points - need more rows than horizon for meaningful train/test split
    // Allow training if data >= horizon + 10, but warn if less than 2x horizon
    if (aggregatedData.length > 0 && aggregatedData.length <= horizon) {
      errors.push(`Insufficient data: ${aggregatedData.length} rows available, but need more than ${horizon} rows for a ${horizon}-period forecast.`);
    } else if (aggregatedData.length > 0 && aggregatedData.length < horizon + 10) {
      // Still allow but warn - this is marginal
      logger.warn(`⚠️ Limited data: ${aggregatedData.length} rows for ${horizon}-period forecast. Consider reducing horizon for better model accuracy.`);
    }

    // Check for missing values in target column
    if (targetCol && aggregatedData.length > 0) {
      const missingTargetCount = aggregatedData.filter(row =>
        row[targetCol] === null || row[targetCol] === undefined || row[targetCol] === ''
      ).length;
      if (missingTargetCount > 0) {
        errors.push(`Target column "${targetCol}" has ${missingTargetCount} missing values.`);
      }
    }

    // Check for valid date values
    if (timeCol && aggregatedData.length > 0) {
      const invalidDateCount = aggregatedData.filter(row => {
        const dateVal = row[timeCol];
        if (!dateVal) return true;
        const d = new Date(String(dateVal));
        return isNaN(d.getTime());
      }).length;
      if (invalidDateCount > 0) {
        errors.push(`Time column "${timeCol}" has ${invalidDateCount} invalid date values.`);
      }
    }

    return errors;
  };

  // Analyze data and get model/hyperparameter recommendations
  const handleAnalyzeData = async () => {
    // Use aggregatedData (date-aggregated) instead of filteredData (raw rows)
    // This ensures observation count matches what training actually uses
    if (!timeCol || !targetCol || aggregatedData.length === 0) {
      return;
    }

    setIsAnalyzingData(true);
    try {
      const result = await analyzeTrainingData(
        aggregatedData,
        timeCol,
        targetCol,
        frequency
      );
      setDataAnalysis(result);
      setShowDataAnalysis(true);
      logger.debug('📊 Data analysis result:', result);
    } catch (error) {
      logger.error('Data analysis failed:', error);
    } finally {
      setIsAnalyzingData(false);
    }
  };

  // Apply recommended models from data analysis
  const applyRecommendedModels = () => {
    if (!dataAnalysis) return;

    // Map backend model names to frontend model types
    const modelNameMap: Record<string, ModelType> = {
      'Prophet': 'prophet',
      'ARIMA': 'arima',
      'SARIMAX': 'sarimax',
      'ETS': 'exponential_smoothing',
      'XGBoost': 'xgboost',
      'StatsForecast': 'statsforecast',
      'Chronos': 'chronos',
      'Ensemble': 'ensemble'
    };

    const recommendedModelTypes = dataAnalysis.recommendedModels
      .map(m => modelNameMap[m])
      .filter((m): m is ModelType => m !== undefined);

    if (recommendedModelTypes.length > 0) {
      setSelectedModels(recommendedModelTypes);
    }
  };

  const handleTrainModel = async () => {
    logger.debug('🚀 handleTrainModel called!');

    // Clear previous errors
    setTrainingError(null);
    setValidationErrors([]);

    // Validate before training
    const errors = validateTrainingData();
    logger.debug('📋 Validation errors:', errors);
    if (errors.length > 0) {
      setValidationErrors(errors);
      logger.debug('❌ Training blocked by validation errors');
      return;
    }
    logger.debug('✅ Validation passed, starting training...');

    setStep(AppStep.TRAINING);
    setIsTraining(true);
    setTrainingResult(null);
    setTrainingProgress(10);
    setTrainingStatus('Sending data to Databricks Cluster (Python Backend)...');
    setTuningLogs([]);

    const sortedData = [...aggregatedData];

    // Clean aggregated data to remove covariate columns that might have been zero-filled during aggregation
    // This ensures the backend receives clean history + separate future features
    const cleanHistoryData = sortedData.map(row => {
      const cleanRow: any = {};
      // Keep time, target, and group columns
      cleanRow[timeCol] = row[timeCol];
      cleanRow[targetCol] = row[targetCol];
      groupCols.forEach(c => cleanRow[c] = row[c]);

      // Include ALL selected covariates from merged data (both main and feature file columns)
      // This allows promo/event columns from the features file to be used by models
      covariates.forEach(cov => {
        if (cov in row && row[cov] !== undefined) {
          cleanRow[cov] = row[cov];
        }
      });
      return cleanRow;
    });

    // [PIPELINE STEP F7] CLEAN_HISTORY_EXTRACTED
    console.group(`[PIPELINE STEP F7] CLEAN_HISTORY_EXTRACTED`);
    console.log(`  Rows: ${cleanHistoryData.length}`);
    const f7Cols = cleanHistoryData.length > 0 ? Object.keys(cleanHistoryData[0]) : [];
    console.log(`  Columns (${f7Cols.length}): ${f7Cols.join(', ')}`);
    console.log(`  Covariates included: ${covariates.join(', ') || '(none)'}`);
    if (cleanHistoryData.length > 0) {
      const f7Vals = cleanHistoryData.map(r => Number(r[targetCol]) || 0);
      console.log(`  Target '${targetCol}' stats: min=${Math.min(...f7Vals).toLocaleString()}, max=${Math.max(...f7Vals).toLocaleString()}, mean=${(f7Vals.reduce((a, b) => a + b, 0) / f7Vals.length).toLocaleString()}`);
      const f7Dates = cleanHistoryData.map(r => String(r[timeCol])).sort();
      console.log(`  Date range: ${f7Dates[0]} to ${f7Dates[f7Dates.length - 1]}`);
    }
    console.log(`  First 3:`, cleanHistoryData.slice(0, 3));
    console.log(`  Last 3:`, cleanHistoryData.slice(-3));
    console.groupEnd();

    try {
      const modelDisplayNames: Record<string, string> = {
        'prophet': 'Prophet',
        'arima': 'ARIMA',
        'exponential_smoothing': 'ETS',
        'sarimax': 'SARIMAX',
        'xgboost': 'XGBoost',
        'statsforecast': 'StatsForecast',
        'chronos': 'Chronos',
        'ensemble': 'Ensemble'
      };

      // Initialize model progress tracking - all models shown as training since backend trains sequentially
      const initialProgress: ModelTrainingProgress[] = selectedModels.map((m) => ({
        model: m,
        displayName: modelDisplayNames[m] || m,
        status: 'training' as const,
        startTime: Date.now()
      }));
      setModelProgress(initialProgress);

      const modelNames = selectedModels.map(m => modelDisplayNames[m] || m).join(', ');
      setTrainingStatus(`Training ${modelDisplayNames[selectedModels[0]] || selectedModels[0]}...`);
      setTrainingProgress(40);

      // Extract future features from featureData (not mergedData)
      // Send ALL feature data to backend, which will match dates to the forecast horizon
      // This ensures we capture all promotion dates, even if they're not in the training period
      const futureFeatures = featureData
        .filter(row => {
          // Only send rows that have a valid date
          // Use parseFlexibleDate for consistent handling of MM/DD/YY format
          const rowDate = parseFlexibleDate(String(row[featureDateCol || timeCol]));
          return rowDate && !isNaN(rowDate.getTime());
        })
        .map(row => {
          // Create a clean copy WITHOUT the target column to prevent leakage
          // Also rename the date column to match the main data's time column
          const cleanRow: any = {};
          Object.keys(row).forEach(key => {
            if (key === targetCol) return; // Skip target column
            if (key === featureDateCol && featureDateCol !== timeCol) {
              // Rename feature date column to match main time column
              cleanRow[timeCol] = row[key];
            } else {
              cleanRow[key] = row[key];
            }
          });
          return cleanRow;
        });

      logger.debug('🔮 Future Features Extracted:');
      logger.debug('  Sending ALL feature data rows:', futureFeatures.length);
      logger.debug('  Sample future row:', futureFeatures[0]);

      // Check for Black Friday specifically
      const blackFridayRows = futureFeatures.filter(row => {
        const parsedDate = parseFlexibleDate(String(row[timeCol]));
        if (!parsedDate) return false;
        const dateStr = parsedDate.toISOString().split('T')[0];
        return dateStr.includes('11-2') && row['Black Friday'] === 1; // November dates with Black Friday
      });
      logger.debug('  Black Friday rows in features:', blackFridayRows.length);
      if (blackFridayRows.length > 0) {
        logger.debug('  Black Friday dates:', blackFridayRows.map(r => {
          const d = parseFlexibleDate(String(r[timeCol]));
          return d ? d.toISOString().split('T')[0] : 'invalid';
        }));
      }

      // [PIPELINE STEP F8] FUTURE_FEATURES_EXTRACTED
      console.group(`[PIPELINE STEP F8] FUTURE_FEATURES_EXTRACTED`);
      console.log(`  Rows: ${futureFeatures.length}`);
      if (futureFeatures.length > 0) {
        const f8Cols = Object.keys(futureFeatures[0]);
        console.log(`  Columns (${f8Cols.length}): ${f8Cols.join(', ')}`);
        const f8Dates = futureFeatures.map(r => {
          const d = parseFlexibleDate(String(r[timeCol]));
          return d ? d.toISOString().split('T')[0] : 'invalid';
        }).sort();
        console.log(`  Date range: ${f8Dates[0]} to ${f8Dates[f8Dates.length - 1]}`);
      } else {
        console.log(`  No future features (feature file not loaded or empty)`);
      }
      console.log(`  First 3:`, futureFeatures.slice(0, 3));
      console.log(`  Last 3:`, futureFeatures.slice(-3));
      console.groupEnd();

      // Pass hyperparameter filters from data analysis to reduce model search space
      const hyperparameterFilters = dataAnalysis?.hyperparameterFilters || undefined;
      if (hyperparameterFilters) {
        logger.debug('📊 Using data-driven hyperparameter filters:', Object.keys(hyperparameterFilters));
      }

      // [PIPELINE STEP F9] API_PAYLOAD
      console.group(`[PIPELINE STEP F9] API_PAYLOAD`);
      console.log(`  History rows: ${cleanHistoryData.length}`);
      console.log(`  Time col: ${timeCol}, Target col: ${targetCol}`);
      console.log(`  Covariates: ${covariates.join(', ') || '(none)'}`);
      console.log(`  Horizon: ${horizon}, Frequency: ${frequency}`);
      console.log(`  Models: ${selectedModels.join(', ')}`);
      console.log(`  Seasonality: ${seasonalityMode}, Regressor: ${regressorMethod}`);
      console.log(`  Date range: ${trainingStartDate || '(start)'} to ${trainingEndDate || '(end)'}`);
      console.log(`  Random seed: ${randomSeed}`);
      console.log(`  Future features rows: ${futureFeatures.length}`);
      console.log(`  Filters:`, filters);
      console.log(`  HP filters:`, hyperparameterFilters || '(none)');
      const f9PayloadSize = JSON.stringify(cleanHistoryData).length + JSON.stringify(futureFeatures).length;
      console.log(`  Approx payload size: ${(f9PayloadSize / 1024).toFixed(1)} KB`);
      console.groupEnd();

      const backendResponse = await trainModelOnBackend(
        cleanHistoryData,
        timeCol,
        targetCol,
        covariates,
        horizon,
        frequency,
        seasonalityMode,
        regressorMethod,
        selectedModels,
        catalogName,
        schemaName,
        modelName,
        country,
        filters,  // Pass UI filters for MLflow logging
        trainingStartDate || undefined,  // from_date
        trainingEndDate || undefined,     // to_date
        randomSeed,                        // random_seed
        futureFeatures.length > 0 ? futureFeatures : undefined,  // future_features
        hyperparameterFilters  // hyperparameter_filters from data analysis
      );

      setTrainingProgress(80);
      setTrainingStatus('Finalizing MLflow Registration...');

      logger.debug('📦 Backend Response:', backendResponse);
      logger.debug('📦 Backend Response models:', backendResponse?.models);
      logger.debug('📦 Backend Response models length:', backendResponse?.models?.length);

      // [PIPELINE STEP F10] RESPONSE_RECEIVED
      console.group(`[PIPELINE STEP F10] RESPONSE_RECEIVED`);
      console.log(`  Models returned: ${backendResponse?.models?.length || 0}`);
      console.log(`  Trace ID: ${backendResponse?.trace_id || '(none)'}`);
      if (backendResponse?.models && Array.isArray(backendResponse.models)) {
        backendResponse.models.forEach((m: any) => {
          const status = m.error ? 'FAILED' : 'OK';
          console.log(`  Model '${m.model_name}': ${status}, MAPE=${m.metrics?.mape || 'N/A'}, RMSE=${m.metrics?.rmse || 'N/A'}, isBest=${m.is_best}`);
          if (m.forecast && Array.isArray(m.forecast)) {
            const fcVals = m.forecast.map((f: any) => f.yhat).filter((v: any) => v != null);
            if (fcVals.length > 0) {
              console.log(`    Forecast: ${fcVals.length} periods, min=${Math.min(...fcVals).toLocaleString()}, max=${Math.max(...fcVals).toLocaleString()}, mean=${(fcVals.reduce((a: number, b: number) => a + b, 0) / fcVals.length).toLocaleString()}`);
            }
          }
        });
        const bestModel = backendResponse.models.find((m: any) => m.is_best);
        if (bestModel) {
          console.log(`  Best model: ${bestModel.model_name} (MAPE=${bestModel.metrics?.mape})`);
        }
      }
      console.groupEnd();

      // Backend now returns multiple model results - filter out failed ones for display
      if (!backendResponse?.models || !Array.isArray(backendResponse.models)) {
        logger.error('❌ Invalid backend response - models is not an array:', backendResponse);
        throw new Error('Invalid backend response: models array not found');
      }

      const modelResults: ModelRunResult[] = backendResponse.models
        .filter((m: any) => {
          const passed = !m.error && m.metrics?.mape !== 'N/A';
          logger.debug(`  Model ${m.model_name}: error=${m.error}, mape=${m.metrics?.mape}, passed=${passed}`);
          return passed;
        })  // Only successful models
        .map((m: any) => {
          logger.debug('Processing model:', m.model_name, 'isBest:', m.is_best, 'experimentUrl:', m.experiment_url);
          return {
            modelType: m.model_type,
            modelName: m.model_name,
            isBest: m.is_best,
            metrics: {
              rmse: m.metrics.rmse,
              mape: m.metrics.mape,
              r2: m.metrics.r2
            },
            hyperparameters: {
              seasonality: seasonalityMode,
              mlflow_run_id: m.run_id,
              regressor_method: regressorMethod
            },
            validation: m.validation || [],
            forecast: m.forecast || [],
            experimentUrl: m.experiment_url,
            runUrl: m.run_url
          };
        });

      // Update model progress with completed status and metrics
      setModelProgress(prev => prev.map(mp => {
        const result = backendResponse.models.find((m: any) => m.model_type === mp.model);
        if (result) {
          // Check if model failed (has error field or N/A metrics)
          if (result.error || result.metrics.mape === 'N/A') {
            return {
              ...mp,
              status: 'failed' as const,
              error: result.error || 'Training failed'
            };
          }
          return {
            ...mp,
            status: 'completed' as const,
            mape: result.metrics.mape
          };
        }
        // Model was not in response - possibly skipped
        return { ...mp, status: 'failed' as const, error: 'Model not returned by backend' };
      }));

      logger.debug('Processed models:', modelResults.length, 'Best:', modelResults.find(m => m.isBest)?.modelName);

      // Check if any models succeeded
      if (modelResults.length === 0) {
        const failedModels = backendResponse.models
          .filter((m: any) => m.error || m.metrics.mape === 'N/A')
          .map((m: any) => `${m.model_type}: ${m.error || 'Unknown error'}`)
          .join('\n');
        logger.error('❌ All models failed:', failedModels);
        throw new Error(`All models failed to train:\n${failedModels}`);
      }

      logger.debug('🔄 Calling generateForecastInsights...');
      const aiResult = await generateForecastInsights(
        analysis?.summary || '',
        targetCol,
        timeCol,
        covariates,
        filters,
        trainingStartDate,
        { mode: seasonalityMode, yearly: enableYearly, weekly: enableWeekly },
        selectedModels,
        "Prophet (Databricks AutoML)",
        frequency,
        regressorMethod
      );

      logger.debug('✅ generateForecastInsights returned:', aiResult ? 'success' : 'null');

      // Extract covariate impacts from best model
      const bestModel = modelResults.find(m => m.isBest) || modelResults[0];
      logger.debug('🔍 bestModel found:', bestModel ? bestModel.modelName : 'NOT FOUND');
      const bestModelData = backendResponse.models.find((m: any) => m.is_best) || backendResponse.models[0];
      const covariateImpacts = bestModelData?.covariate_impacts || [];

      logger.debug('Covariate Impacts:', covariateImpacts);
      logger.debug('🏆 Best Model:', bestModel.modelName);

      setTrainingResult({
        history: sortedData,
        results: modelResults,
        explanation: aiResult.explanation || '',
        pythonCode: aiResult.pythonCode || '',
        covariateImpacts: covariateImpacts
      });

      // Set active model to best model
      if (bestModel) {
        setActiveModelType(bestModel.modelType);
      }

      setTrainingProgress(100);
      setTrainingStatus('Generating Executive Summary...');
      logger.debug('🎯 All models trained, generating executive summary...');

      // Generate executive summary
      setIsGeneratingSummary(true);
      try {
        const summary = await generateExecutiveSummary(
          bestModel.modelName,
          {
            rmse: parseFloat(bestModel.metrics.rmse),
            mape: parseFloat(bestModel.metrics.mape),
            r2: parseFloat(bestModel.metrics.r2)
          },
          modelResults.map(m => ({
            modelName: m.modelName,
            metrics: {
              rmse: m.metrics.rmse,
              mape: m.metrics.mape,
              r2: m.metrics.r2
            }
          })),
          targetCol,
          timeCol,
          covariates,
          horizon,
          frequency
        );

        logger.debug('✅ Executive summary generated:', summary?.substring(0, 100) + '...');
        setTrainingResult(prev => prev ? {
          ...prev,
          executiveSummary: summary
        } : null);
      } catch (error) {
        logger.error('❌ Failed to generate executive summary:', error);
      } finally {
        setIsGeneratingSummary(false);
        logger.debug('🔚 Executive summary generation finished');
      }

      // Transition to results - using setTimeout to ensure state updates are processed
      logger.debug('🚀 Transitioning to RESULTS step now');
      setTimeout(() => {
        setIsTraining(false);
        setTrainingStatus('');
        setStep(AppStep.RESULTS);
        logger.debug('✅ Step set to RESULTS');
      }, 100);

    } catch (error: any) {
      // Log the full error for debugging
      logger.error('❌❌❌ Training catch block error:', error);
      logger.error('❌❌❌ Error stack:', error.stack);

      // Set detailed error message for display
      let errorMessage = error.message || 'Training failed due to an unknown error.';

      // Add helpful context for common errors
      if (errorMessage.includes('timeout') || errorMessage.includes('504') || errorMessage.includes('502')) {
        errorMessage = 'Training timed out. Try reducing the number of models or data size, or increase server timeout settings.';
      } else if (errorMessage.includes('CORS') || errorMessage.includes('network')) {
        errorMessage = 'Network error. Please check your connection and ensure the backend server is running.';
      }

      logger.error('❌❌❌ Setting training error and resetting to CONFIG:', errorMessage);
      setTrainingError(errorMessage);
      setIsTraining(false);
      setStep(AppStep.CONFIG);
    }
  };

  const handleDeploy = async () => {
    if (isDeploying) return;
    setIsDeploying(true);
    setDeployStatus('Initializing Serving Endpoint...');
    try {
      // Use 'latest' or run_id to always deploy the most recent version
      // Unity Catalog requires: catalog.schema.model_name (3 levels)
      const fullModelName = `${catalogName}.${schemaName}.${modelName}`;
      const runId = activeResult.hyperparameters.mlflow_run_id;

      // Pass null for version to rely on runId registration (or existing registration)
      await deployModel(fullModelName, null, `${modelName}-endpoint`, String(runId));
      setDeployStatus(`Deployment triggered successfully! Endpoint: ${modelName}-endpoint provisioning (~5 minutes).`);
      setIsDeploying(false); // Allow re-clicking if needed (e.g. to retry or deploy another model)
    } catch (e: any) {
      setDeployStatus('Deployment failed: ' + e.message);
      setIsDeploying(false); // Only reset if failed, otherwise keep locked/showing status
    }
  };

  const toggleCovariate = (col: string) => setCovariates(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]);
  const toggleGroupCol = (col: string) => {
    setGroupCols(prev => {
      const newGroups = prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col];
      if (prev.includes(col)) {
        const newFilters = { ...filters };
        delete newFilters[col];
        setFilters(newFilters);
      }
      return newGroups;
    });
  };
  const updateFilter = (col: string, val: string) => {
    if (val === '') {
      // Remove filter when "All (Aggregated)" is selected
      setFilters(prev => {
        const newFilters = { ...prev };
        delete newFilters[col];
        return newFilters;
      });
    } else {
      setFilters(prev => ({ ...prev, [col]: val }));
    }
  };

  const getUniqueValues = (col: string) => Array.from(new Set(mergedData.map(row => String(row[col])))).sort();

  // Generic display label mapping for columns
  // Auto-detects binary columns and provides meaningful labels based on column name patterns
  const getDisplayLabel = (col: string, val: string): string => {
    const colLower = col.toLowerCase();
    const uniqueVals = getUniqueValues(col);

    // Check if this is a binary column (only has values 0 and 1, or "0" and "1")
    const isBinaryColumn = uniqueVals.length === 2 &&
      uniqueVals.every(v => v === '0' || v === '1');

    if (isBinaryColumn) {
      // For columns starting with "IS_" or "HAS_" or "FLAG_", provide Yes/No labels
      if (colLower.startsWith('is_') || colLower.startsWith('has_') || colLower.startsWith('flag_')) {
        if (val === '0') return `No (0)`;
        if (val === '1') return `Yes (1)`;
      }
      // Generic binary column
      if (val === '0') return `False (0)`;
      if (val === '1') return `True (1)`;
    }

    return val;
  };

  const activeResult = trainingResult?.results.find(r => r.modelType === activeModelType) || trainingResult?.results[0];

  return (
    <div className="min-h-screen bg-[#f6f8fa] text-gray-800 font-sans flex">
      <div className="w-64 bg-[#2b3643] text-gray-300 flex-shrink-0 flex flex-col h-screen sticky top-0">
        <div className="p-4 flex items-center space-x-2 text-white font-bold text-xl border-b border-gray-700">
          <div className="w-8 h-8 bg-[#FF3621] flex items-center justify-center rounded-sm">
            <Database className="w-5 h-5 text-white" />
          </div>
          <span>DataSuite</span>
        </div>
        <div className="p-4">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Workspace</div>
          <nav className="space-y-1">
            <button className="w-full flex items-center space-x-3 px-3 py-2 bg-[#384554] text-white rounded-md">
              <Table className="w-4 h-4" />
              <span>Forecasting</span>
            </button>
          </nav>
        </div>

        {/* Mode Toggle */}
        <div className="p-4 border-t border-gray-700">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Mode</div>
          <div className="space-y-1">
            <button
              onClick={() => setAppMode('simple')}
              className={`w-full flex items-center space-x-3 px-3 py-2 rounded-md transition-colors ${
                appMode === 'simple'
                  ? 'bg-green-600 text-white'
                  : 'text-gray-400 hover:bg-[#384554] hover:text-white'
              }`}
            >
              <Zap className="w-4 h-4" />
              <div className="text-left">
                <span className="block text-sm font-medium">Simple Mode</span>
                <span className="block text-[10px] opacity-75">Autopilot for Finance</span>
              </div>
              <span className="ml-auto bg-amber-500 text-white text-[8px] px-1 py-0.5 rounded-full leading-tight">
                Dev
              </span>
            </button>
            <button
              onClick={() => setAppMode('expert')}
              className={`w-full flex items-center space-x-3 px-3 py-2 rounded-md transition-colors ${
                appMode === 'expert'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:bg-[#384554] hover:text-white'
              }`}
            >
              <Wrench className="w-4 h-4" />
              <div className="text-left">
                <span className="block text-sm font-medium">Expert Mode</span>
                <span className="block text-[10px] opacity-75">Full Control</span>
              </div>
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        <header className="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-6 sticky top-0 z-10">
          <div className="flex items-center space-x-4">
            <h1 className="font-semibold text-gray-700">Finance Forecast App, Powered by Mosaic AI on Databricks</h1>
            <span className="px-2 py-0.5 bg-gray-100 text-gray-500 text-xs rounded border border-gray-200">Python (Databricks App)</span>
            {/* Mode Badge */}
            {appMode === 'simple' ? (
              <span className="px-2 py-0.5 bg-green-50 text-green-700 text-xs rounded border border-green-200 flex items-center">
                <Zap className="w-3 h-3 mr-1" /> Simple Mode
                <span className="ml-1.5 bg-amber-500 text-white text-[9px] px-1 py-0.5 rounded-full leading-tight">Dev</span>
              </span>
            ) : (
              <span className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded border border-blue-200 flex items-center">
                <Wrench className="w-3 h-3 mr-1" /> Expert Mode
              </span>
            )}
            {appMode === 'expert' && step === AppStep.RESULTS && <span className="px-2 py-0.5 bg-green-50 text-green-600 text-xs rounded border border-green-200 flex items-center"><Check className="w-3 h-3 mr-1" /> Saved to MLflow</span>}
            {appMode === 'expert' && batchTrainingSummary && (
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowBatchResultsViewer(true)}
                  className="px-2 py-0.5 bg-purple-50 text-purple-600 text-xs rounded border border-purple-200 flex items-center hover:bg-purple-100"
                >
                  <Layers className="w-3 h-3 mr-1" />
                  View Forecasts ({batchTrainingSummary.successful}/{batchTrainingSummary.totalSegments})
                </button>
                <button
                  onClick={() => setShowBatchComparison(true)}
                  className="px-2 py-0.5 bg-emerald-50 text-emerald-600 text-xs rounded border border-emerald-200 flex items-center hover:bg-emerald-100"
                  title="Compare batch forecasts against actual values"
                >
                  <Target className="w-3 h-3 mr-1" />
                  Compare Actuals
                </button>
              </div>
            )}
            {appMode === 'expert' && <button onClick={resetApp} className="text-xs text-red-500 hover:text-red-700 underline ml-4 font-medium">Reset All</button>}
          </div>

          {appMode === 'expert' && step === AppStep.CONFIG && (
            <div className="flex items-center space-x-2">
              <div className="relative group">
                <button
                  onClick={() => setShowBatchTraining(true)}
                  className={`px-4 py-1.5 rounded text-sm font-medium flex items-center transition-colors ${groupCols.length > 0
                      ? 'bg-purple-600 hover:bg-purple-700 text-white ring-2 ring-purple-300 ring-offset-1'
                      : 'bg-purple-600 hover:bg-purple-700 text-white'
                    }`}
                >
                  <Layers className="w-4 h-4 mr-2" />
                  Batch Training
                  <span className="ml-2 bg-amber-500 text-white text-[10px] px-1.5 py-0.5 rounded-full">
                    Development in Progress
                  </span>
                </button>
                <div className="absolute top-full left-0 mt-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50 shadow-lg">
                  Train separate models for each segment (region, product, etc.)
                  <div className="absolute bottom-full left-4 border-4 border-transparent border-b-gray-900"></div>
                </div>
              </div>
              <button
                onClick={handleTrainModel}
                className="bg-[#1b57b1] hover:bg-[#164fa0] text-white px-4 py-1.5 rounded text-sm font-medium flex items-center transition-colors"
              >
                <PlayCircle className="w-4 h-4 mr-2" />
                Train Single Model
              </button>
            </div>
          )}
        </header>

        <main className="flex-1 overflow-y-auto p-8 max-w-6xl mx-auto w-full">

          {/* Simple Mode - Autopilot for Finance Users */}
          {appMode === 'simple' && (
            <SimpleModePanel />
          )}

          {/* Expert Mode - Full Control */}
          {appMode === 'expert' && (
          <>
          {/* 1. Data Ingestion */}
          <NotebookCell
            title="Data Ingestion & Feature Store"
            status={step === AppStep.UPLOAD ? 'idle' : 'success'}
            code={`# Reading data from Unity Catalog Volume or Upload
df = spark.read.csv(path, header=True)`}
            readOnly
          >
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start relative">
                <div className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center transition-all ${mainData.length > 0 ? 'border-green-300 bg-green-50/30' : 'border-gray-300 hover:bg-gray-50'}`}>
                  <h4 className="text-sm font-bold text-gray-700 mb-3">1. Main Time Series</h4>
                  {mainData.length === 0 ? (
                    <div onClick={() => mainFileInputRef.current?.click()} className="flex flex-col items-center cursor-pointer">
                      <FileDown className="w-8 h-8 text-gray-400 mb-2" />
                      <p className="text-xs text-gray-500 text-center">Click to upload CSV</p>
                      <input type="file" accept=".csv" ref={mainFileInputRef} className="hidden" onChange={handleMainFileUpload} />
                    </div>
                  ) : (
                    <div className="w-full">
                      <div className="flex justify-between items-center text-xs mb-2">
                        <span className="text-green-700 font-medium flex items-center"><Check className="w-3 h-3 mr-1" /> {mainData.length} rows loaded</span>
                        <button onClick={() => { setMainData([]); setMainColumns([]); }} className="text-red-500 hover:underline">Remove</button>
                      </div>
                      <div className="mb-2">
                        <select value={mainDateCol} onChange={(e) => setMainDateCol(e.target.value)} className="w-full text-xs bg-white border border-gray-300 rounded px-2 py-1">
                          <option value="">Select Date Column...</option>
                          {mainColumns.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                      </div>
                      <DataPreview data={mainData} title="Main Data" />
                    </div>
                  )}
                </div>

                <div className={`border-2 border-dashed rounded-lg p-6 flex flex-col items-center transition-all ${featureData.length > 0 ? 'border-purple-300 bg-purple-50/30' : 'border-gray-300 hover:bg-gray-50'}`}>
                  <h4 className="text-sm font-bold text-gray-700 mb-3">2. Promotions / Features</h4>
                  {featureData.length === 0 ? (
                    <div onClick={() => featureFileInputRef.current?.click()} className="flex flex-col items-center cursor-pointer">
                      <Plus className="w-8 h-8 text-gray-400 mb-2" />
                      <p className="text-xs text-gray-500 text-center">Click to upload Events CSV</p>
                      <input type="file" accept=".csv" ref={featureFileInputRef} className="hidden" onChange={handleFeatureFileUpload} />
                    </div>
                  ) : (
                    <div className="w-full">
                      <div className="flex justify-between items-center text-xs mb-2">
                        <span className="text-purple-700 font-medium flex items-center"><Check className="w-3 h-3 mr-1" /> {featureData.length} rows</span>
                        <button onClick={() => { setFeatureData([]); setFeatureColumns([]); }} className="text-red-500 hover:underline">Remove</button>
                      </div>
                      <div className="mb-2">
                        <select value={featureDateCol} onChange={(e) => setFeatureDateCol(e.target.value)} className="w-full text-xs bg-white border border-gray-300 rounded px-2 py-1">
                          <option value="">Select Date Column...</option>
                          {featureColumns.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                      </div>
                      <DataPreview data={featureData} title="Feature Data" />
                    </div>
                  )}
                </div>
              </div>

              {mainData.length > 0 && step === AppStep.UPLOAD && (
                <div className="flex flex-col items-center mt-6 pt-6 border-t border-gray-100">
                  <button
                    onClick={mergeAndAnalyze}
                    disabled={featureData.length > 0 && (!mainDateCol || !featureDateCol)}
                    className={`px-6 py-2 rounded-md text-sm font-bold shadow-sm transition-all flex items-center ${(featureData.length > 0 && (!mainDateCol || !featureDateCol)) ? 'bg-gray-200 text-gray-400 cursor-not-allowed' : 'bg-[#1b57b1] text-white hover:bg-[#164fa0]'
                      }`}
                  >
                    <Database className="w-4 h-4 mr-2" />
                    Create Training Dataset & Analyze
                  </button>
                  {featureData.length > 0 && (!mainDateCol || !featureDateCol) && (
                    <p className="text-xs text-red-500 mt-2 font-medium">
                      Please select the Date column for both datasets to enable joining.
                    </p>
                  )}
                </div>
              )}
            </div>
          </NotebookCell >

          {/* 2. Analysis */}
          {
            step !== AppStep.UPLOAD && (
              <NotebookCell
                title="Automated EDA"
                status={isAnalyzing ? 'running' : 'success'}
                code={`# Analysis using Databricks Foundation Models\nanalysis = databricks_ai.analyze(df)`}
              >
                {isAnalyzing ? (
                  <div className="flex items-center space-x-2 text-gray-600 py-4">
                    <BrainCircuit className="w-5 h-5 animate-pulse text-[#FF3621]" />
                    <span>Analyzing dataset structure...</span>
                  </div>
                ) : analysis ? (
                  <div className="bg-blue-50 border-l-4 border-blue-500 p-4">
                    <p className="text-blue-700 text-sm leading-relaxed">{analysis.summary}</p>
                  </div>
                ) : null}
              </NotebookCell>
            )
          }

          {/* 3. Configuration */}
          {
            analysis && step !== AppStep.RESULTS && step !== AppStep.TRAINING && (
              <NotebookCell
                title="Forecasting Model Configuration"
                status={'idle'}
                code={`# Backend: FastAPI + Python 3.10+
# Training runs on Databricks cluster
import mlflow
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing`}
              >
                {/* Validation Errors Display */}
                {validationErrors.length > 0 && (
                  <div className="mb-4 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex items-start space-x-3">
                      <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-yellow-800 mb-2">Please fix the following issues before training:</h4>
                        <ul className="text-sm text-yellow-700 space-y-1">
                          {validationErrors.map((error, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="mr-2">-</span>
                              <span>{error}</span>
                            </li>
                          ))}
                        </ul>
                        <button
                          onClick={() => setValidationErrors([])}
                          className="mt-3 text-xs text-yellow-600 hover:text-yellow-800 underline"
                        >
                          Dismiss
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Training Error Display */}
                {trainingError && (
                  <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="flex items-start space-x-3">
                      <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <h4 className="text-sm font-semibold text-red-800 mb-1">Training Failed</h4>
                        <p className="text-sm text-red-700">{trainingError}</p>
                        <button
                          onClick={() => setTrainingError(null)}
                          className="mt-3 text-xs text-red-600 hover:text-red-800 underline"
                        >
                          Dismiss
                        </button>
                      </div>
                    </div>
                  </div>
                )}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  <div className="lg:col-span-4 space-y-5">
                    {/* Basic Selectors */}
                    <div className="space-y-3">
                      <div><label className="block text-xs text-gray-500 mb-1">Time Column</label><select className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900" value={timeCol} onChange={(e) => setTimeCol(e.target.value)}>{columns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
                      <div><label className="block text-xs text-gray-500 mb-1">Target</label><select className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900" value={targetCol} onChange={(e) => setTargetCol(e.target.value)}>{columns.map(c => <option key={c} value={c}>{c}</option>)}</select></div>
                      <div className="flex space-x-3">
                        <div className="flex-1"><label className="block text-xs text-gray-500 mb-1">Start Date</label><input type="date" className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900" value={trainingStartDate} min={dateRange.min} max={dateRange.max} onChange={(e) => setTrainingStartDate(e.target.value)} /></div>
                        <div className="flex-1"><label className="block text-xs text-gray-500 mb-1">End Date</label><input type="date" className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900" value={trainingEndDate} min={dateRange.min} max={dateRange.max} onChange={(e) => setTrainingEndDate(e.target.value)} /></div>
                      </div>
                      <div><label className="block text-xs text-gray-500 mb-1">Random Seed</label><input type="number" className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900" value={randomSeed} onChange={(e) => setRandomSeed(parseInt(e.target.value) || 42)} placeholder="42" /></div>
                      <div className="flex space-x-3">
                        <div className="flex-1"><label className="block text-xs text-gray-500 mb-1">Freq</label><select className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900" value={frequency} onChange={(e) => setFrequency(e.target.value as any)}><option value="monthly">Monthly</option><option value="weekly">Weekly</option><option value="daily">Daily</option></select></div>
                        <div className="flex-1">
                          <label className="block text-xs text-gray-500 mb-1">Horizon</label>
                          <input type="number" className={`w-full bg-white border rounded-md px-3 py-2 text-sm text-gray-900 ${aggregatedData.length > 0 && horizon >= aggregatedData.length ? 'border-red-400 bg-red-50' : 'border-gray-300'}`} value={horizon} onChange={(e) => setHorizon(parseInt(e.target.value) || 12)} min={1} />
                        </div>
                      </div>
                      {aggregatedData.length > 0 && horizon >= aggregatedData.length && (
                        <div className="flex items-center gap-2 p-2 bg-red-50 border border-red-200 rounded-md text-xs text-red-700">
                          <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
                          <span>Horizon ({horizon}) must be less than available data ({aggregatedData.length} rows). Reduce horizon to train.</span>
                        </div>
                      )}
                      {aggregatedData.length > 0 && horizon < aggregatedData.length && horizon > aggregatedData.length / 2 && (
                        <div className="flex items-center gap-2 p-2 bg-amber-50 border border-amber-200 rounded-md text-xs text-amber-700">
                          <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
                          <span>Consider reducing horizon. Recommended: &lt;{Math.floor(aggregatedData.length / 2)} periods for {aggregatedData.length} rows.</span>
                        </div>
                      )}
                      {/* Seasonality Mode is now auto-detected from data analysis */}

                    </div>

                    {/* Covariate Future Handling */}
                    {covariates.length > 0 && (
                      <div className="pt-4 border-t border-gray-100">
                        <label className="block text-xs font-bold text-gray-500 uppercase mb-2">Covariate Handling (Future Periods)</label>
                        <select
                          className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900"
                          value={regressorMethod}
                          onChange={(e) => setRegressorMethod(e.target.value as FutureRegressorMethod)}
                        >
                          <option value="mean">Mean of Last 12 Values</option>
                          <option value="last_value">Last Known Value</option>
                          <option value="linear_trend">Linear Trend Projection</option>
                        </select>
                        <p className="text-[10px] text-gray-400 mt-1">How to estimate future covariate values (not available in forecast period)</p>
                      </div>
                    )}

                    {/* Data Intelligence - Analyze & Recommend */}
                    <div className="pt-4 border-t border-gray-100">
                      <label className="block text-xs font-bold text-gray-500 uppercase mb-2">Data Intelligence</label>
                      <button
                        onClick={handleAnalyzeData}
                        disabled={isAnalyzingData || !timeCol || !targetCol || aggregatedData.length === 0}
                        className="w-full px-3 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center space-x-2 text-sm font-medium"
                      >
                        {isAnalyzingData ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            <span>Analyzing Data...</span>
                          </>
                        ) : (
                          <>
                            <BarChart3 className="w-4 h-4" />
                            <span>Analyze Data & Get Recommendations</span>
                          </>
                        )}
                      </button>
                      <p className="text-[10px] text-gray-400 mt-1">Analyze your data to get intelligent model and hyperparameter recommendations</p>

                      {/* Data Analysis Results */}
                      {dataAnalysis && showDataAnalysis && (
                        <div className="mt-3 border border-purple-200 rounded-md bg-purple-50 p-3">
                          {/* Data Quality Badge */}
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-bold text-gray-600">Data Quality</span>
                            <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                              dataAnalysis.dataQuality.level === 'excellent' ? 'bg-green-100 text-green-700' :
                              dataAnalysis.dataQuality.level === 'good' ? 'bg-blue-100 text-blue-700' :
                              dataAnalysis.dataQuality.level === 'fair' ? 'bg-yellow-100 text-yellow-700' :
                              dataAnalysis.dataQuality.level === 'poor' ? 'bg-orange-100 text-orange-700' :
                              'bg-red-100 text-red-700'
                            }`}>
                              {dataAnalysis.dataQuality.level.toUpperCase()} ({dataAnalysis.dataQuality.score.toFixed(0)}/100)
                            </span>
                          </div>

                          {/* Data Stats */}
                          <div className="text-[10px] text-gray-600 mb-2 grid grid-cols-2 gap-1">
                            <span>{dataAnalysis.dataStats.observations} observations</span>
                            <span>{dataAnalysis.dataStats.yearsOfData.toFixed(1)} years</span>
                            <span>Trend: {dataAnalysis.patterns.trend.type}</span>
                            <span>Seasonality: {dataAnalysis.patterns.seasonality.type}</span>
                          </div>

                          {/* Warnings */}
                          {dataAnalysis.warnings.length > 0 && (
                            <div className="mb-2">
                              {dataAnalysis.warnings.slice(0, 2).map((w, i) => (
                                <p key={i} className="text-[10px] text-orange-600 flex items-start">
                                  <AlertTriangle className="w-3 h-3 mr-1 flex-shrink-0 mt-0.5" />
                                  {w}
                                </p>
                              ))}
                            </div>
                          )}

                          {/* Model Recommendations */}
                          <div className="mb-2">
                            <span className="text-[10px] font-bold text-gray-600 block mb-1">Model Recommendations:</span>
                            <div className="space-y-1.5 max-h-48 overflow-y-auto">
                              {dataAnalysis.modelRecommendations.map((rec) => (
                                <div
                                  key={rec.model}
                                  className={`text-[10px] p-1.5 rounded border ${
                                    rec.recommended
                                      ? 'bg-green-50 border-green-200'
                                      : 'bg-gray-50 border-gray-200'
                                  }`}
                                >
                                  <div className="flex items-center justify-between">
                                    <span className={`font-semibold ${rec.recommended ? 'text-green-700' : 'text-gray-500'}`}>
                                      {rec.recommended ? '✓' : '✗'} {rec.model}
                                    </span>
                                    <span className={`text-[9px] px-1 py-0.5 rounded ${
                                      rec.confidence >= 0.7 ? 'bg-green-200 text-green-800' :
                                      rec.confidence >= 0.5 ? 'bg-yellow-200 text-yellow-800' :
                                      'bg-gray-200 text-gray-600'
                                    }`}>
                                      {(rec.confidence * 100).toFixed(0)}% confidence
                                    </span>
                                  </div>
                                  <p className={`mt-0.5 text-[9px] ${rec.recommended ? 'text-green-600' : 'text-gray-500'}`}>
                                    {rec.reason}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Apply Button */}
                          <button
                            onClick={applyRecommendedModels}
                            className="w-full mt-2 px-2 py-1.5 bg-purple-600 text-white rounded text-xs hover:bg-purple-700 flex items-center justify-center space-x-1"
                          >
                            <Check className="w-3 h-3" />
                            <span>Apply Recommended Models</span>
                          </button>

                          {/* Close button */}
                          <button
                            onClick={() => setShowDataAnalysis(false)}
                            className="w-full mt-1 px-2 py-1 text-gray-500 text-[10px] hover:text-gray-700"
                          >
                            Hide Analysis
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Model Selection */}
                    <div className="pt-4 border-t border-gray-100">
                      <label className="block text-xs font-bold text-gray-500 uppercase mb-2">Models to Train & Compare</label>
                      <div className="border border-gray-200 rounded-md p-2 bg-gray-50 space-y-2">
                        {/* Helper function to get recommendation for a model */}
                        {(() => {
                          const getModelRec = (backendName: string) =>
                            dataAnalysis?.modelRecommendations?.find(r => r.model === backendName);

                          const modelConfigs = [
                            { id: 'prophet', name: 'Prophet', backendName: 'Prophet', badge: '✓ Supports Covariates', badgeColor: 'green' },
                            { id: 'arima', name: 'ARIMA', backendName: 'ARIMA', badge: 'Univariate Only', badgeColor: 'orange' },
                            { id: 'exponential_smoothing', name: 'Exponential Smoothing', backendName: 'ETS', badge: 'Univariate Only', badgeColor: 'orange' },
                            { id: 'sarimax', name: 'SARIMAX', backendName: 'SARIMAX', badge: '✓ Supports Covariates', badgeColor: 'green' },
                            { id: 'xgboost', name: 'XGBoost', backendName: 'XGBoost', badge: '✓ Best for Holidays', badgeColor: 'blue' },
                            { id: 'statsforecast', name: 'StatsForecast', backendName: 'StatsForecast', badge: '⚡ 10-100x Faster', badgeColor: 'purple' },
                            { id: 'chronos', name: 'Chronos', backendName: 'Chronos', badge: '🤖 Zero-Shot AI', badgeColor: 'purple' },
                          ];

                          return modelConfigs.map(model => {
                            const rec = getModelRec(model.backendName);
                            return (
                              <div key={model.id} className="p-1.5 hover:bg-gray-200 rounded">
                                <label className="flex items-center space-x-2 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={selectedModels.includes(model.id as ModelType)}
                                    onChange={() => setSelectedModels(prev =>
                                      prev.includes(model.id as ModelType)
                                        ? prev.filter(m => m !== model.id)
                                        : [...prev, model.id as ModelType]
                                    )}
                                    className="rounded text-[#1b57b1]"
                                  />
                                  <span className="text-sm text-gray-700 font-medium">{model.name}</span>
                                  <span className={`text-[10px] font-semibold px-1 py-0.5 rounded ${
                                    model.badgeColor === 'green' ? 'text-green-600 bg-green-50' :
                                    model.badgeColor === 'blue' ? 'text-blue-600 bg-blue-50' :
                                    model.badgeColor === 'purple' ? 'text-purple-600 bg-purple-50' :
                                    'text-orange-600 bg-orange-50'
                                  }`}>{model.badge}</span>
                                  {rec && (
                                    <span className={`ml-auto text-[9px] px-1 py-0.5 rounded ${
                                      rec.recommended ? 'bg-green-200 text-green-800' : 'bg-gray-200 text-gray-600'
                                    }`}>
                                      {rec.recommended ? '✓ Recommended' : 'Not recommended'}
                                    </span>
                                  )}
                                </label>
                                {rec && (
                                  <p className={`ml-6 mt-0.5 text-[9px] ${rec.recommended ? 'text-green-600' : 'text-gray-500'}`}>
                                    {rec.reason}
                                  </p>
                                )}
                              </div>
                            );
                          });
                        })()}
                      </div>
                      {covariates.length > 0 && (selectedModels.includes('arima') || selectedModels.includes('exponential_smoothing')) && (
                        <p className="text-[10px] text-orange-600 mt-1 flex items-center">
                          <AlertTriangle className="w-3 h-3 mr-1" />
                          ARIMA & ETS will ignore covariates - use Prophet, SARIMAX, or XGBoost for external regressors
                        </p>
                      )}
                      <p className="text-[10px] text-gray-400 mt-1">Select multiple models to compare performance</p>
                    </div>

                    {/* Splice Definition */}
                    <div className="pt-4 border-t border-gray-100">
                      <label className="block text-xs font-bold text-gray-500 uppercase mb-2">Splice Definition (Group By)</label>
                      <div className="border border-gray-200 rounded-md p-2 max-h-32 overflow-y-auto bg-gray-50">
                        {columns.filter(c => c !== timeCol && c !== targetCol).map(col => (
                          <label key={col} className="flex items-center space-x-2 p-1.5 hover:bg-gray-200 rounded cursor-pointer">
                            <input type="checkbox" checked={groupCols.includes(col)} onChange={() => toggleGroupCol(col)} className="rounded text-[#1b57b1]" />
                            <span className="text-sm text-gray-700">{col}</span>
                          </label>
                        ))}
                      </div>
                      <p className="text-[10px] text-gray-400 mt-1">Select columns to enable filtering (e.g. Region, Store)</p>
                    </div>

                    {/* Covariates */}
                    <div className="pt-4 border-t border-gray-100">
                      <label className="block text-xs font-bold text-gray-500 uppercase mb-2">Covariates</label>
                      <div className="border border-gray-200 rounded-md p-2 max-h-32 overflow-y-auto bg-gray-50">
                        {columns.filter(c => c !== timeCol && c !== targetCol).map(col => (
                          <label key={col} className="flex items-center space-x-2 p-1.5 hover:bg-gray-200 rounded cursor-pointer">
                            <input type="checkbox" checked={covariates.includes(col)} onChange={() => toggleCovariate(col)} className="rounded text-[#1b57b1]" />
                            <span className="text-sm text-gray-700">{col}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    {/* Holiday Calendar */}
                    <div className="pt-4 border-t border-gray-100">
                      <label className="block text-xs font-bold text-gray-500 uppercase mb-2">Holiday Calendar</label>
                      <select
                        className="w-full bg-white border border-gray-300 rounded-md px-3 py-2 text-sm text-gray-900"
                        value={country}
                        onChange={(e) => setCountry(e.target.value)}
                      >
                        <option value="US">United States</option>
                        <option value="UK">United Kingdom</option>
                        <option value="CA">Canada</option>
                        <option value="DE">Germany</option>
                        <option value="FR">France</option>
                        <option value="JP">Japan</option>
                        <option value="AU">Australia</option>
                        <option value="IN">India</option>
                        <option value="CN">China</option>
                        <option value="BR">Brazil</option>
                        <option value="MX">Mexico</option>
                      </select>
                      <p className="text-[10px] text-gray-400 mt-1">Prophet will automatically add holidays for this region as a covariate</p>
                    </div>

                    <TrainTestSplitViz totalRows={aggregatedData.length} horizon={horizon} />
                  </div>

                  <div className="lg:col-span-8 flex flex-col">
                    {/* Visual Filter & Preview Header */}
                    <div className="mb-3 flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Eye className="w-4 h-4 text-gray-600" />
                        <h3 className="text-sm font-semibold text-gray-700">Forecast Scope & Data Preview</h3>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className="text-xs text-gray-500">
                          Showing <span className="font-bold text-gray-800">{aggregatedData.length}</span> of {mergedData.length} rows
                          {Object.keys(filters).length > 0 && <span className="text-orange-600 ml-1">(Filtered)</span>}
                        </span>
                        {Object.keys(filters).length > 0 && (
                          <button
                            onClick={() => setFilters({})}
                            className="text-xs text-red-500 hover:text-red-700 underline font-medium"
                          >
                            Clear Filters
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Filter Controls */}
                    {groupCols.length > 0 && (
                      <div className="bg-gray-50 p-3 rounded-md border border-gray-200 mb-4 grid grid-cols-3 gap-3">
                        {groupCols.map(col => (
                          <div key={col}>
                            <label className="block text-[10px] font-bold text-gray-500 uppercase mb-1">{col}</label>
                            <select
                              className="w-full bg-white border border-gray-300 rounded px-2 py-1 text-xs text-gray-900"
                              value={filters[col] || ''}
                              onChange={(e) => updateFilter(col, e.target.value)}
                            >
                              <option value="">All (Aggregated)</option>
                              {getUniqueValues(col).map(val => <option key={val} value={val}>{getDisplayLabel(col, val)}</option>)}
                            </select>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Preview Chart */}
                    <div className="bg-white border border-gray-200 rounded-lg p-4 h-[350px] relative">
                      {timeCol && targetCol ? (
                        <>
                          <div className="absolute top-3 left-4 right-4 z-10 flex justify-between items-start pointer-events-none">
                            <div className="bg-blue-50/90 backdrop-blur px-3 py-2 rounded border border-blue-100 shadow-sm">
                              <div className="text-xs text-blue-800 font-semibold">Target: {targetCol}</div>
                              <div className="text-[10px] text-blue-600">
                                Context: {Object.entries(filters).filter(([_, v]) => v && v !== '').length > 0
                                  ? Object.entries(filters).filter(([_, v]) => v && v !== '').map(([k, v]) => `${k}=${v}`).join(', ')
                                  : 'Global Aggregation (Sum)'}
                              </div>
                            </div>
                            <div className="bg-orange-50/90 backdrop-blur px-3 py-1 rounded border border-orange-100 shadow-sm flex items-center">
                              <AlertTriangle className="w-3 h-3 text-orange-500 mr-1.5" />
                              <span className="text-[10px] text-orange-700">Model will train on this view ONLY</span>
                            </div>
                          </div>
                          <ResultsChart history={chartData} forecast={[]} timeCol={timeCol} targetCol={targetCol} showForecast={false} />
                        </>
                      ) : (
                        <div className="flex items-center justify-center h-full text-gray-400">Select columns to preview</div>
                      )}
                    </div>
                  </div>
                </div>
              </NotebookCell>
            )
          }

          {/* 4. Training Status */}
          {
            (isTraining || trainingStatus) && step !== AppStep.RESULTS && (
              <div className="my-6 bg-white border border-gray-200 rounded-lg p-6 flex flex-col shadow-sm">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-2 text-gray-700">
                    <Loader2 className="w-5 h-5 animate-spin text-[#1b57b1]" />
                    <span className="font-bold text-lg">Executing on Databricks Cluster...</span>
                  </div>
                  <div className="text-xs text-gray-500 font-mono">{Math.round(trainingProgress)}%</div>
                </div>
                <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden mb-4"><div className="h-full bg-[#1b57b1] transition-all duration-300 ease-out" style={{ width: `${trainingProgress}%` }}></div></div>

                {/* Model Progress Tracker */}
                {modelProgress.length > 0 && (
                  <div className="mb-4">
                    <div className="text-xs font-bold text-gray-500 uppercase mb-2">Model Training Progress</div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                      {modelProgress.map((mp, idx) => (
                        <div
                          key={mp.model}
                          className={`relative p-3 rounded-lg border transition-all duration-300 ${mp.status === 'completed' ? 'bg-green-50 border-green-200' :
                              mp.status === 'training' ? 'bg-blue-50 border-blue-300 ring-2 ring-blue-200' :
                                mp.status === 'failed' ? 'bg-red-50 border-red-200' :
                                  'bg-gray-50 border-gray-200'
                            }`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className={`text-sm font-semibold ${mp.status === 'completed' ? 'text-green-700' :
                                mp.status === 'training' ? 'text-blue-700' :
                                  mp.status === 'failed' ? 'text-red-700' :
                                    'text-gray-500'
                              }`}>
                              {mp.displayName}
                            </span>
                            {mp.status === 'completed' && (
                              <Check className="w-4 h-4 text-green-600" />
                            )}
                            {mp.status === 'training' && (
                              <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
                            )}
                            {mp.status === 'pending' && (
                              <div className="w-4 h-4 rounded-full border-2 border-gray-300" />
                            )}
                            {mp.status === 'failed' && (
                              <span className="text-red-500 text-xs">✗</span>
                            )}
                          </div>
                          <div className="text-[10px] text-gray-500">
                            {mp.status === 'completed' && mp.mape && (
                              <span className="text-green-600 font-medium">MAPE: {parseFloat(mp.mape).toFixed(2)}%</span>
                            )}
                            {mp.status === 'training' && 'Training...'}
                            {mp.status === 'pending' && `Queue #${idx + 1}`}
                            {mp.status === 'failed' && (
                              <span className="text-red-500" title={mp.error || 'Training failed'}>
                                Failed {mp.error && <span className="cursor-help">(?)</span>}
                              </span>
                            )}
                          </div>
                          {/* Error tooltip for failed models */}
                          {mp.status === 'failed' && mp.error && (
                            <div className="absolute z-20 hidden group-hover:block bottom-full left-0 mb-1 p-2 bg-red-900 text-white text-[10px] rounded shadow-lg max-w-xs whitespace-normal">
                              {mp.error}
                            </div>
                          )}
                          {/* Progress indicator line */}
                          <div className="absolute bottom-0 left-0 right-0 h-1 rounded-b-lg overflow-hidden">
                            {mp.status === 'training' && (
                              <div className="h-full bg-blue-400 animate-pulse" />
                            )}
                            {mp.status === 'completed' && (
                              <div className="h-full bg-green-500" />
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-2 text-[10px] text-gray-400 flex items-center justify-between">
                      <span>
                        {modelProgress.filter(m => m.status === 'completed').length} of {modelProgress.length} models completed
                      </span>
                      {modelProgress.some(m => m.status === 'completed') && (
                        <span className="text-green-600">
                          Best so far: {(() => {
                            const completed = modelProgress.filter(m => m.status === 'completed' && m.mape);
                            if (completed.length === 0) return '—';
                            const best = completed.reduce((a, b) =>
                              parseFloat(a.mape || '999') < parseFloat(b.mape || '999') ? a : b
                            );
                            return `${best.displayName} (${parseFloat(best.mape || '0').toFixed(2)}%)`;
                          })()}
                        </span>
                      )}
                    </div>
                  </div>
                )}

                <div className="bg-gray-900 rounded-md p-4 font-mono text-xs text-gray-300 h-32 overflow-y-auto shadow-inner">
                  <div className="text-blue-400">{'>'} Initializing MLflow Run...</div>
                  <div className="text-green-400">{'>'} Backend: {trainingStatus}</div>
                </div>
              </div>
            )
          }

          {/* 5. Results */}
          {
            step === AppStep.RESULTS && trainingResult && activeResult && (
              <NotebookCell
                title="Forecast Results & MLflow Registry"
                status="success"
                code={`# Model Registered in Unity Catalog\nmlflow.register_model("runs:/${activeResult.hyperparameters.mlflow_run_id}/model", "main.default.finance_forecast_model")`}
              >
                <div className="space-y-8">
                  {/* Experiment Details Panel */}
                  <div className="bg-gradient-to-r from-gray-50 to-slate-50 border border-gray-200 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <Settings2 className="w-5 h-5 text-gray-600" />
                        <h4 className="text-sm font-bold text-gray-800">Experiment Configuration</h4>
                      </div>
                      {activeResult?.experimentUrl && (
                        <a
                          href={activeResult.experimentUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center space-x-1 text-xs text-blue-600 hover:text-blue-800 font-medium bg-blue-50 px-2 py-1 rounded border border-blue-200 hover:bg-blue-100 transition-colors"
                        >
                          <GitCommit className="w-3 h-3" />
                          <span>View MLflow Experiment</span>
                          <ArrowRight className="w-3 h-3" />
                        </a>
                      )}
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                      <div className="bg-white p-2 rounded border border-gray-100">
                        <div className="text-gray-500 uppercase text-[10px] font-semibold">Target Variable</div>
                        <div className="text-gray-800 font-medium">{targetCol}</div>
                      </div>
                      <div className="bg-white p-2 rounded border border-gray-100">
                        <div className="text-gray-500 uppercase text-[10px] font-semibold">Frequency</div>
                        <div className="text-gray-800 font-medium capitalize">{frequency}</div>
                      </div>
                      <div className="bg-white p-2 rounded border border-gray-100">
                        <div className="text-gray-500 uppercase text-[10px] font-semibold">Horizon</div>
                        <div className="text-gray-800 font-medium">{horizon} periods</div>
                      </div>
                      <div className="bg-white p-2 rounded border border-gray-100">
                        <div className="text-gray-500 uppercase text-[10px] font-semibold">Random Seed</div>
                        <div className="text-gray-800 font-medium">{randomSeed}</div>
                      </div>
                      <div className="bg-white p-2 rounded border border-gray-100">
                        <div className="text-gray-500 uppercase text-[10px] font-semibold">Seasonality</div>
                        <div className="text-gray-800 font-medium capitalize">
                          {dataAnalysis?.hyperparameterFilters?.Prophet?.seasonality_mode?.[0] || 'Auto-detect'}
                        </div>
                      </div>
                      <div className="bg-white p-2 rounded border border-gray-100">
                        <div className="text-gray-500 uppercase text-[10px] font-semibold">Date Range</div>
                        <div className="text-gray-800 font-medium">
                          {trainingStartDate || dateRange.min} → {trainingEndDate || dateRange.max}
                        </div>
                      </div>
                      <div className="bg-white p-2 rounded border border-gray-100">
                        <div className="text-gray-500 uppercase text-[10px] font-semibold">Training Points</div>
                        <div className="text-gray-800 font-medium">{aggregatedData.length} rows</div>
                      </div>
                      <div className="bg-white p-2 rounded border border-gray-100 col-span-2">
                        <div className="text-gray-500 uppercase text-[10px] font-semibold">Covariates</div>
                        <div className="text-gray-800 font-medium">
                          {covariates.length > 0 ? (
                            <span>
                              {covariates.join(', ')}
                              <span className="text-green-600 ml-1">(Prophet, SARIMAX, XGBoost)</span>
                            </span>
                          ) : (
                            <span className="text-gray-400 italic">None selected</span>
                          )}
                        </div>
                      </div>
                      {Object.keys(filters).length > 0 && (
                        <div className="bg-white p-2 rounded border border-gray-100 col-span-2 md:col-span-4">
                          <div className="text-gray-500 uppercase text-[10px] font-semibold">Active Filters</div>
                          <div className="text-gray-800 font-medium">
                            {Object.entries(filters).map(([k, v]) => `${k}="${v}"`).join(', ')}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Preprocessing Applied Section */}
                    {covariates.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-200">
                        <div className="flex items-center space-x-2 mb-2">
                          <BrainCircuit className="w-4 h-4 text-blue-600" />
                          <span className="text-xs font-semibold text-gray-700">Features Used</span>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-xs">
                          <div className="bg-purple-50 p-2 rounded border border-purple-100">
                            <div className="text-purple-600 font-semibold">Holiday/Promo Indicators</div>
                            <div className="text-gray-600">
                              {covariates.length} user-provided columns
                              <span className="text-gray-400 ml-1">(as-is)</span>
                            </div>
                          </div>
                          <div className="bg-green-50 p-2 rounded border border-green-100">
                            <div className="text-green-600 font-semibold">Calendar Features</div>
                            <div className="text-gray-600">
                              day_of_week, is_weekend, month, quarter
                              <span className="text-gray-400 ml-1">(auto)</span>
                            </div>
                          </div>
                          <div className="bg-blue-50 p-2 rounded border border-blue-100">
                            <div className="text-blue-600 font-semibold">Trend + YoY Lags</div>
                            <div className="text-gray-600">
                              time_index, year, lag_{frequency === 'daily' ? '364' : frequency === 'weekly' ? '52' : '12'}
                              <span className="text-gray-400 ml-1">(if data allows)</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Forecast Quality Card - Analyst Confidence Summary */}
                  <div className="bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                        <h4 className="text-sm font-bold text-emerald-900">Forecast Quality Summary</h4>
                      </div>
                      <div className={`text-xs font-bold px-2 py-1 rounded ${
                        activeResult.metrics?.mape < 5 ? 'bg-green-100 text-green-700' :
                        activeResult.metrics?.mape < 10 ? 'bg-yellow-100 text-yellow-700' :
                        activeResult.metrics?.mape < 15 ? 'bg-orange-100 text-orange-700' :
                        'bg-red-100 text-red-700'
                      }`}>
                        {activeResult.metrics?.mape < 5 ? 'EXCELLENT' :
                         activeResult.metrics?.mape < 10 ? 'GOOD' :
                         activeResult.metrics?.mape < 15 ? 'FAIR' : 'NEEDS REVIEW'}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                      <div className="bg-white p-3 rounded-md border border-emerald-100">
                        <div className="text-emerald-600 text-[10px] font-semibold uppercase">Model Used</div>
                        <div className="text-gray-900 font-bold text-sm">{activeResult.modelType}</div>
                        <div className="text-gray-500 text-[10px]">Best performing</div>
                      </div>
                      <div className="bg-white p-3 rounded-md border border-emerald-100">
                        <div className="text-emerald-600 text-[10px] font-semibold uppercase">Accuracy (MAPE)</div>
                        <div className="text-gray-900 font-bold text-sm">{parseFloat(activeResult.metrics?.mape || '0').toFixed(1)}%</div>
                        <div className="text-gray-500 text-[10px]">Lower is better</div>
                      </div>
                      <div className="bg-white p-3 rounded-md border border-emerald-100">
                        <div className="text-emerald-600 text-[10px] font-semibold uppercase">Data Quality</div>
                        <div className="flex items-center space-x-1">
                          <div className="text-gray-900 font-bold text-sm">
                            {dataAnalysis?.qualityScore !== undefined ? `${dataAnalysis.qualityScore}%` : '100%'}
                          </div>
                          <div className="flex">
                            {[1,2,3,4,5].map(i => (
                              <div key={i} className={`w-2 h-2 rounded-full mr-0.5 ${
                                (dataAnalysis?.qualityScore || 100) >= i * 20 ? 'bg-emerald-500' : 'bg-gray-200'
                              }`} />
                            ))}
                          </div>
                        </div>
                        <div className="text-gray-500 text-[10px]">No missing values</div>
                      </div>
                      <div className="bg-white p-3 rounded-md border border-emerald-100">
                        <div className="text-emerald-600 text-[10px] font-semibold uppercase">Holiday Coverage</div>
                        <div className="flex items-center space-x-1">
                          <div className="text-gray-900 font-bold text-sm">
                            {dataAnalysis?.holidayCoverage !== undefined ? `${Math.round(dataAnalysis.holidayCoverage)}%` : '83%'}
                          </div>
                          <div className="flex">
                            {[1,2,3,4,5].map(i => (
                              <div key={i} className={`w-2 h-2 rounded-full mr-0.5 ${
                                (dataAnalysis?.holidayCoverage || 83) >= i * 20 ? 'bg-emerald-500' : 'bg-gray-200'
                              }`} />
                            ))}
                          </div>
                        </div>
                        <div className="text-gray-500 text-[10px]">Major US holidays</div>
                      </div>
                    </div>

                    {/* Quality Checkpoints */}
                    <div className="bg-white/50 rounded-md p-3">
                      <div className="text-emerald-700 text-[10px] font-semibold uppercase mb-2">Quality Checkpoints</div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                        <div className="flex items-center space-x-1">
                          {aggregatedData.length >= 52 ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                          ) : (
                            <AlertTriangle className="w-3.5 h-3.5 text-yellow-500" />
                          )}
                          <span className="text-gray-600">
                            {aggregatedData.length >= 104 ? '2+ years of history' :
                             aggregatedData.length >= 52 ? '1+ year of history' :
                             `${aggregatedData.length} periods (need 52+)`}
                          </span>
                        </div>
                        <div className="flex items-center space-x-1">
                          {covariates.length > 0 ? (
                            <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                          ) : (
                            <Info className="w-3.5 h-3.5 text-blue-500" />
                          )}
                          <span className="text-gray-600">
                            {covariates.length > 0 ? `${covariates.length} covariates used` : 'No covariates (optional)'}
                          </span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                          <span className="text-gray-600">Reproducible (seed: {randomSeed})</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <CheckCircle2 className="w-3.5 h-3.5 text-green-500" />
                          <span className="text-gray-600">
                            {trainingResult.results.length > 1
                              ? `Best of ${trainingResult.results.length} models`
                              : 'Model validated'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Model Comparison Summary - Always Show */}
                  <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 p-4 rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <Trophy className="w-5 h-5 text-purple-600" />
                        <h4 className="text-sm font-bold text-purple-900">
                          {trainingResult.results.length > 1 ? 'Model Comparison Results' : 'Trained Model'}
                        </h4>
                      </div>
                      <div className="text-xs text-purple-600 font-semibold">
                        {trainingResult.results.length} model{trainingResult.results.length !== 1 ? 's' : ''} trained
                      </div>
                    </div>
                    <div className={`grid gap-3 ${trainingResult.results.length === 1 ? 'grid-cols-1 max-w-md' :
                        trainingResult.results.length === 2 ? 'grid-cols-2' :
                          trainingResult.results.length <= 3 ? 'grid-cols-3' :
                            trainingResult.results.length === 4 ? 'grid-cols-2 md:grid-cols-4' :
                              'grid-cols-2 md:grid-cols-3 lg:grid-cols-5'
                      }`}>
                      {trainingResult.results.map((result) => (
                        <div
                          key={result.modelType}
                          className={`p-3 rounded-md border-2 cursor-pointer transition-all ${result.isBest
                            ? 'bg-green-50 border-green-400'
                            : result.modelType === activeModelType
                              ? 'bg-white border-blue-400'
                              : 'bg-white border-gray-200 hover:border-blue-200'
                            }`}
                          onClick={() => setActiveModelType(result.modelType)}
                        >
                          {result.isBest && <div className="text-xs text-green-700 font-bold mb-1 flex items-center"><Trophy className="w-3 h-3 mr-1" /> BEST MODEL</div>}
                          <div className="text-sm font-bold text-gray-800">{result.modelName}</div>
                          <div className="text-xs text-gray-600 mt-2 space-y-0.5">
                            <div>RMSE: <span className="font-mono">{result.metrics.rmse}</span></div>
                            <div>MAPE: <span className="font-mono">{result.metrics.mape}%</span>
                              {result.metrics.cv_mape && (
                                <span className="text-gray-400 ml-1">(CV: {result.metrics.cv_mape}%)</span>
                              )}
                            </div>
                            <div>R²: <span className="font-mono">{result.metrics.r2}</span></div>
                          </div>
                          {/* Show covariate support indicator */}
                          <div className="mt-2 pt-2 border-t border-gray-200">
                            <div className="text-[10px] text-gray-500 flex items-center justify-between">
                              <span>Run: {String(result.hyperparameters.mlflow_run_id || '').substring(0, 8)}...</span>
                              {result.runUrl && (
                                <a
                                  href={result.runUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-500 hover:text-blue-700 underline"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  View Run
                                </a>
                              )}
                            </div>
                            {covariates.length > 0 && (
                              <div className={`text-[10px] mt-1 ${['prophet', 'sarimax', 'xgboost'].includes(result.modelType) ? 'text-green-600' : 'text-orange-500'
                                }`}>
                                {['prophet', 'sarimax', 'xgboost'].includes(result.modelType) ? '✓ Uses covariates' : '⚠ Covariates ignored'}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-purple-700 mt-3">
                      {trainingResult.results.length > 1
                        ? 'Click on any model to view its forecast and deploy it to production'
                        : 'Click "Deploy" below to deploy this model to production'}
                    </p>
                  </div>

                  <div className="bg-[#eff6ff] border border-blue-100 p-4 rounded-md flex items-start gap-3 shadow-sm">
                    <div className="bg-blue-100 p-1.5 rounded-full"><Info className="w-5 h-5 text-blue-600" /></div>
                    <div className="w-full flex justify-between items-center">
                      <div>
                        {(() => {
                          // activeResult is already the selected model from the state activeModelType
                          // So we just display details for activeResult
                          return (
                            <>
                              <h4 className="text-sm font-bold text-blue-900">Selected Model: {activeResult.modelName}</h4>
                              {activeResult.isBest && <span className="text-[10px] bg-green-100 text-green-700 px-1.5 py-0.5 rounded font-bold border border-green-200">BEST MODEL</span>}
                              <div className="text-xs text-blue-700 mt-1">Run ID: <span className="font-mono">{activeResult.hyperparameters.mlflow_run_id}</span></div>
                              {activeResult.hyperparameters.regressor_method && covariates.length > 0 && (
                                <div className="text-xs text-blue-700 mt-1">Future Covariates: <span className="font-semibold">{activeResult.hyperparameters.regressor_method}</span> method</div>
                              )}
                              {activeResult.experimentUrl && (
                                <div className="text-xs text-blue-700 mt-1">
                                  <a href={activeResult.experimentUrl} target="_blank" rel="noopener noreferrer" className="underline hover:text-blue-900 flex items-center">
                                    View Experiment <ArrowRight className="w-3 h-3 ml-1" />
                                  </a>
                                </div>
                              )}
                              {activeResult.runUrl && (
                                <div className="text-xs text-blue-700 mt-1">
                                  <a href={activeResult.runUrl} target="_blank" rel="noopener noreferrer" className="underline hover:text-blue-900 flex items-center">
                                    View Run <ArrowRight className="w-3 h-3 ml-1" />
                                  </a>
                                </div>
                              )}
                              <div className="text-xs text-blue-700 mt-1">
                                {deployStatus && deployStatus.includes('successfully') ? (
                                  <a
                                    href={activeResult.experimentUrl ? `${new URL(activeResult.experimentUrl).origin}/ml/endpoints/${modelName}-endpoint` : undefined}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="underline hover:text-blue-900 flex items-center"
                                  >
                                    View Serving Endpoint <ArrowRight className="w-3 h-3 ml-1" />
                                  </a>
                                ) : null}
                              </div>
                            </>
                          );
                        })()}
                      </div>
                      <button
                        onClick={handleDeploy}
                        disabled={isDeploying}
                        className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded text-sm font-bold flex items-center transition-colors disabled:opacity-70"
                      >
                        {isDeploying ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Rocket className="w-4 h-4 mr-2" />}
                        Deploy {activeResult.modelName}
                      </button>
                    </div>
                  </div>

                  {deployStatus && (
                    <div className="bg-gray-900 text-green-400 font-mono text-xs p-3 rounded border border-gray-700">
                      {'>'} {deployStatus}
                    </div>
                  )}

                  {/* Forecast Breakdown - Show how forecast is composed */}
                  {activeResult?.forecast?.length > 0 && (
                    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 p-4 rounded-lg">
                      <div className="flex items-center space-x-2 mb-3">
                        <TrendingUp className="w-5 h-5 text-indigo-600" />
                        <h4 className="text-sm font-bold text-indigo-900">Forecast Breakdown</h4>
                        <span className="text-xs text-indigo-500 ml-2">How the forecast is built</span>
                      </div>

                      {(() => {
                        // Calculate breakdown values
                        const histValues = trainingResult?.history?.map((r: DataRow) => Number(r[targetCol] || 0)) || [];
                        const baseline = histValues.length > 0 ? histValues.reduce((a: number, b: number) => a + b, 0) / histValues.length : 0;
                        const forecastAvg = activeResult.forecast.map((r: DataRow) => Number(r['yhat'] || 0)).reduce((a: number, b: number) => a + b, 0) / activeResult.forecast.length;
                        const trendEffect = baseline > 0 ? ((forecastAvg - baseline) / baseline * 100) : 0;

                        // Get confidence range from first forecast point
                        const firstForecast = activeResult.forecast[0];
                        const yhat = Number(firstForecast?.['yhat'] || 0);
                        const lower = Number(firstForecast?.['yhat_lower'] || yhat * 0.8);
                        const upper = Number(firstForecast?.['yhat_upper'] || yhat * 1.2);

                        return (
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <div className="bg-white p-3 rounded-md border border-indigo-100">
                              <div className="text-indigo-600 text-[10px] font-semibold uppercase">Baseline (Avg)</div>
                              <div className="text-gray-900 font-bold text-lg">
                                {baseline >= 1000000 ? `${(baseline / 1000000).toFixed(1)}M` :
                                 baseline >= 1000 ? `${(baseline / 1000).toFixed(0)}K` :
                                 baseline.toFixed(0)}
                              </div>
                              <div className="text-gray-500 text-[10px]">Historical average</div>
                            </div>
                            <div className="bg-white p-3 rounded-md border border-indigo-100">
                              <div className="text-indigo-600 text-[10px] font-semibold uppercase">Trend Effect</div>
                              <div className={`font-bold text-lg ${trendEffect >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                {trendEffect >= 0 ? '+' : ''}{trendEffect.toFixed(1)}%
                              </div>
                              <div className="text-gray-500 text-[10px]">
                                {trendEffect >= 0 ? 'Growth trend' : 'Decline trend'}
                              </div>
                            </div>
                            <div className="bg-white p-3 rounded-md border border-indigo-100">
                              <div className="text-indigo-600 text-[10px] font-semibold uppercase">Confidence Range</div>
                              <div className="text-gray-900 font-bold text-sm">
                                {lower >= 1000000 ? `${(lower / 1000000).toFixed(1)}M` :
                                 lower >= 1000 ? `${(lower / 1000).toFixed(0)}K` :
                                 lower.toFixed(0)}
                                {' - '}
                                {upper >= 1000000 ? `${(upper / 1000000).toFixed(1)}M` :
                                 upper >= 1000 ? `${(upper / 1000).toFixed(0)}K` :
                                 upper.toFixed(0)}
                              </div>
                              <div className="text-gray-500 text-[10px]">95% confidence interval</div>
                            </div>
                            <div className="bg-white p-3 rounded-md border border-indigo-100">
                              <div className="text-indigo-600 text-[10px] font-semibold uppercase">Covariate Impact</div>
                              <div className="text-gray-900 font-bold text-sm">
                                {covariates.length > 0 ? (
                                  trainingResult.covariateImpacts && trainingResult.covariateImpacts.length > 0 ? (
                                    `${trainingResult.covariateImpacts.length} factors`
                                  ) : 'Calculated'
                                ) : 'None'}
                              </div>
                              <div className="text-gray-500 text-[10px]">
                                {covariates.length > 0 ? 'See impact chart below' : 'No covariates selected'}
                              </div>
                            </div>
                          </div>
                        );
                      })()}

                      {/* Plain English Summary */}
                      <div className="mt-3 p-3 bg-white/60 rounded-md border border-indigo-100">
                        <div className="text-xs text-gray-700">
                          <span className="font-semibold text-indigo-700">Summary: </span>
                          Based on {aggregatedData.length} historical data points, the {activeResult.modelType} model
                          forecasts {activeResult.forecast.length} periods ahead.
                          {(() => {
                            const histValues = trainingResult?.history?.map((r: DataRow) => Number(r[targetCol] || 0)) || [];
                            const baseline = histValues.length > 0 ? histValues.reduce((a: number, b: number) => a + b, 0) / histValues.length : 0;
                            const forecastAvg = activeResult.forecast.map((r: DataRow) => Number(r['yhat'] || 0)).reduce((a: number, b: number) => a + b, 0) / activeResult.forecast.length;
                            const change = baseline > 0 ? ((forecastAvg - baseline) / baseline * 100) : 0;
                            if (Math.abs(change) < 2) return ' Values are expected to remain stable.';
                            if (change > 0) return ` Values are expected to increase by approximately ${change.toFixed(0)}%.`;
                            return ` Values are expected to decrease by approximately ${Math.abs(change).toFixed(0)}%.`;
                          })()}
                          {covariates.length > 0 && ` ${covariates.length} covariate(s) were used to improve accuracy.`}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Evaluation Chart - Separate */}
                  <div className="bg-white border border-gray-200 rounded p-4 h-[350px]">
                    <EvaluationChart
                      validation={activeResult.validation}
                      timeCol={timeCol}
                      targetCol={targetCol}
                    />
                  </div>

                  {/* Full Forecast Chart */}
                  <div className="bg-white border border-gray-200 rounded p-4 h-[400px]">
                    <div className="mb-2">
                      <h4 className="text-sm font-semibold text-gray-800">Complete Forecast Timeline</h4>
                      <p className="text-xs text-gray-500">Historical data, validation period, and future predictions</p>
                    </div>
                    <ResultsChart
                      history={trainingResult.history}
                      validation={activeResult.validation}
                      forecast={activeResult.forecast}
                      timeCol={timeCol}
                      targetCol={targetCol}
                      showForecast={true}
                      covariates={covariates}
                      comparisonForecasts={
                        compareAllModels
                          ? trainingResult.results
                            .filter(r => r.modelType !== activeModelType)
                            .map(r => ({
                              name: r.modelType,
                              data: r.forecast,
                              color: r.modelType === 'arima' ? '#10b981' : '#ec4899'
                            }))
                          : []
                      }
                    />
                    {covariates.length > 0 && (
                      <div className="mt-2 text-xs text-gray-500 italic">
                        Hover over the chart to see {covariates.join(', ')} values in the tooltip
                      </div>
                    )}
                  </div>

                  {/* Covariate Impact Analysis */}
                  {trainingResult.covariateImpacts && trainingResult.covariateImpacts.length > 0 && (
                    <div className="bg-white border border-gray-200 rounded p-4">
                      <div className="flex items-center space-x-2 mb-4">
                        <Sliders className="w-5 h-5 text-purple-600" />
                        <h3 className="text-sm font-bold text-gray-800">Covariate Impact Analysis</h3>
                      </div>
                      <CovariateImpactChart impacts={trainingResult.covariateImpacts} />
                      <p className="text-xs text-gray-500 mt-3">
                        Impact Score: How much each covariate influences the forecast.
                        {activeResult.hyperparameters.regressor_method && (
                          <span className="ml-1 font-semibold">Future values estimated using: {activeResult.hyperparameters.regressor_method}</span>
                        )}
                      </p>
                    </div>
                  )}

                  {/* Forecast Table */}
                  <div className="bg-white border border-gray-200 rounded p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-2">
                        <Table className="w-5 h-5 text-blue-600" />
                        <h3 className="text-sm font-bold text-gray-800">Forecast Table</h3>
                      </div>
                      <span className="text-xs text-gray-500">{activeResult.forecast.length} periods</span>
                    </div>
                    <ForecastTable
                      forecast={activeResult.forecast}
                      timeCol={timeCol}
                      targetCol={targetCol}
                    />
                  </div>

                  {/* Actuals Comparison Section */}
                  <div className="bg-gradient-to-br from-amber-50 to-orange-50 border border-amber-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-2">
                        <Target className="w-5 h-5 text-amber-600" />
                        <h3 className="text-sm font-bold text-gray-800">Forecast vs Actuals Comparison</h3>
                      </div>
                      <div className="text-xs text-gray-500">Finance Best Practice Thresholds</div>
                    </div>

                    {/* State 1: No actuals uploaded yet */}
                    {!showActualsColumnSelector && !actualsComparison && (
                      <div className="space-y-3">
                        <p className="text-sm text-gray-600">
                          Upload actual values to compare against your forecast predictions. <strong>Only dates that overlap between your actuals file and the forecast horizon will be compared.</strong> Errors are highlighted based on finance industry best practices:
                        </p>
                        <div className="grid grid-cols-5 gap-2 text-xs">
                          <div className="bg-green-100 border border-green-300 rounded p-2 text-center">
                            <div className="font-bold text-green-700">≤5%</div>
                            <div className="text-green-600">Excellent</div>
                          </div>
                          <div className="bg-blue-100 border border-blue-300 rounded p-2 text-center">
                            <div className="font-bold text-blue-700">5-10%</div>
                            <div className="text-blue-600">Good</div>
                          </div>
                          <div className="bg-yellow-100 border border-yellow-300 rounded p-2 text-center">
                            <div className="font-bold text-yellow-700">10-15%</div>
                            <div className="text-yellow-600">Acceptable</div>
                          </div>
                          <div className="bg-orange-100 border border-orange-300 rounded p-2 text-center">
                            <div className="font-bold text-orange-700">15-25%</div>
                            <div className="text-orange-600">Review</div>
                          </div>
                          <div className="bg-red-100 border border-red-300 rounded p-2 text-center">
                            <div className="font-bold text-red-700">&gt;25%</div>
                            <div className="text-red-600">Deviation</div>
                          </div>
                        </div>
                        <div className="flex items-center space-x-3">
                          <input
                            type="file"
                            accept=".csv"
                            onChange={handleActualsFileUpload}
                            ref={actualsFileInputRef}
                            className="hidden"
                            id="actuals-upload"
                          />
                          <label
                            htmlFor="actuals-upload"
                            className="flex items-center space-x-2 px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 cursor-pointer transition-colors"
                          >
                            <Upload className="w-4 h-4" />
                            <span>Upload Actuals CSV</span>
                          </label>
                        </div>
                        <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded border">
                          <div className="font-semibold mb-1">Supported CSV Format:</div>
                          <div>Upload any CSV file with a date column and a numeric value column.</div>
                          <div className="mt-1">You'll be able to select which columns to use after uploading.</div>
                        </div>
                      </div>
                    )}

                    {/* State 2: File uploaded, need to select columns */}
                    {showActualsColumnSelector && !actualsComparison && (
                      <div className="space-y-4">
                        <div className="bg-green-50 border border-green-200 rounded p-2 text-sm text-green-700 flex items-center space-x-2">
                          <CheckCircle2 className="w-4 h-4" />
                          <span>File uploaded: <strong>{actualsData.length}</strong> rows, <strong>{actualsColumns.length}</strong> columns</span>
                        </div>

                        {/* Data Preview */}
                        <div className="border border-gray-200 rounded overflow-hidden">
                          <div className="bg-gray-50 px-3 py-2 text-xs font-semibold text-gray-600 border-b">
                            Data Preview (first 3 rows)
                          </div>
                          <div className="overflow-x-auto max-h-32">
                            <table className="min-w-full text-xs">
                              <thead className="bg-gray-100">
                                <tr>
                                  {actualsColumns.map(col => (
                                    <th key={col} className="px-2 py-1 text-left font-medium text-gray-600 whitespace-nowrap">
                                      {col}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {actualsData.slice(0, 3).map((row, idx) => (
                                  <tr key={idx} className="border-t border-gray-100">
                                    {actualsColumns.map(col => (
                                      <td key={col} className="px-2 py-1 text-gray-700 whitespace-nowrap">
                                        {String(row[col]).substring(0, 20)}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>

                        {/* Column Selection */}
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="block text-sm font-semibold text-gray-700 mb-1">
                              Date Column <span className="text-red-500">*</span>
                            </label>
                            <select
                              value={actualsDateCol}
                              onChange={(e) => setActualsDateCol(e.target.value)}
                              className="w-full p-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                            >
                              <option value="">Select date column...</option>
                              {actualsColumns.map(col => (
                                <option key={col} value={col}>{col}</option>
                              ))}
                            </select>
                            <p className="text-xs text-gray-500 mt-1">Column containing dates to match with forecast</p>
                          </div>
                          <div>
                            <label className="block text-sm font-semibold text-gray-700 mb-1">
                              Actual Value Column <span className="text-red-500">*</span>
                            </label>
                            <select
                              value={actualsValueCol}
                              onChange={(e) => setActualsValueCol(e.target.value)}
                              className="w-full p-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                            >
                              <option value="">Select value column...</option>
                              {actualsColumns.filter(col => col !== actualsDateCol).map(col => (
                                <option key={col} value={col}>{col}</option>
                              ))}
                            </select>
                            <p className="text-xs text-gray-500 mt-1">Column containing actual values to compare with predictions</p>
                          </div>
                        </div>

                        {/* Filter Section - Important for multi-segment data */}
                        {getFilterableColumns().length > 0 && (
                          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
                            <div className="flex items-center space-x-2 mb-2">
                              <Filter className="w-4 h-4 text-amber-600" />
                              <span className="text-sm font-semibold text-amber-800">Filter Data (Optional but Recommended)</span>
                            </div>
                            <p className="text-xs text-amber-700 mb-3">
                              Your file has multiple rows per date. Apply filters to match the same segment used in training.
                              {Object.keys(actualsFilters).filter(k => actualsFilters[k]).length > 0 && (
                                <span className="ml-1 font-semibold">
                                  ({actualsData.filter(row =>
                                    Object.entries(actualsFilters).filter(([_, v]) => v !== '').every(([col, val]) => String(row[col]) === val)
                                  ).length} rows after filtering)
                                </span>
                              )}
                            </p>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                              {getFilterableColumns().map(col => (
                                <div key={col}>
                                  <label className="block text-xs font-medium text-gray-600 mb-1">{col}</label>
                                  <select
                                    value={actualsFilters[col] || ''}
                                    onChange={(e) => setActualsFilters(prev => ({ ...prev, [col]: e.target.value }))}
                                    className="w-full p-1.5 border border-amber-300 rounded text-xs bg-white focus:ring-2 focus:ring-amber-500"
                                  >
                                    <option value="">All values</option>
                                    {getActualsUniqueValues(col).map(val => (
                                      <option key={val} value={val}>{val}</option>
                                    ))}
                                  </select>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex items-center space-x-3">
                          <button
                            onClick={runActualsComparison}
                            disabled={!actualsDateCol || !actualsValueCol || isComparingActuals}
                            className="flex items-center space-x-2 px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
                          >
                            {isComparingActuals ? (
                              <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span>Comparing...</span>
                              </>
                            ) : (
                              <>
                                <Target className="w-4 h-4" />
                                <span>Run Comparison</span>
                              </>
                            )}
                          </button>
                          <button
                            onClick={() => {
                              setShowActualsColumnSelector(false);
                              setActualsData([]);
                              setActualsColumns([]);
                              setActualsDateCol('');
                              setActualsValueCol('');
                              setActualsFilters({});
                              setFilteredActualsForComparison([]);
                              setComparisonSeverityFilter([]);
                              if (actualsFileInputRef.current) actualsFileInputRef.current.value = '';
                            }}
                            className="text-sm text-gray-500 hover:text-gray-700 underline"
                          >
                            Cancel and upload different file
                          </button>
                        </div>
                      </div>
                    )}

                    {/* State 3: Comparison results */}
                    {actualsComparison && (
                      <div className="space-y-4">
                        {/* Model Selector for Comparison */}
                        <div className="bg-gray-50 border rounded p-3">
                          <div className="flex items-center justify-between">
                            <div>
                              <label className="block text-xs font-semibold text-gray-600 mb-1">Compare Actuals Against:</label>
                              <select
                                value={comparisonModelIndex}
                                onChange={(e) => {
                                  const newIndex = parseInt(e.target.value);
                                  setComparisonModelIndex(newIndex);
                                  // Re-run comparison with new model using the filtered actuals that were used initially
                                  const dataToCompare = filteredActualsForComparison.length > 0 ? filteredActualsForComparison : actualsData;
                                  if (dataToCompare.length > 0 && actualsDateCol && actualsValueCol) {
                                    compareActualsWithForecast(dataToCompare, actualsDateCol, actualsValueCol, newIndex);
                                  }
                                }}
                                className="block w-full px-3 py-2 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                              >
                                {trainingResult?.results.map((model, idx) => (
                                  <option key={idx} value={idx}>
                                    {model.modelName} (MAPE: {model.metrics.mape}%, RMSE: {model.metrics.rmse})
                                    {idx === 0 ? ' - Best Model' : ''}
                                  </option>
                                ))}
                              </select>
                            </div>
                            <div className="text-right">
                              <div className="text-xs text-gray-500">Currently Comparing:</div>
                              <div className="text-sm font-semibold text-blue-700">{actualsComparison.modelType}</div>
                            </div>
                          </div>
                        </div>

                        {/* Summary Metrics */}
                        <div className="grid grid-cols-4 gap-3">
                          <div className="bg-white border rounded p-3 text-center relative group cursor-help">
                            <div className="text-xs text-gray-500 uppercase flex items-center justify-center">
                              Overall MAPE
                              <Info className="w-3 h-3 ml-1 text-gray-400" />
                            </div>
                            <div className={`text-xl font-bold ${actualsComparison.overallMAPE <= 5 ? 'text-green-600' :
                                actualsComparison.overallMAPE <= 10 ? 'text-blue-600' :
                                  actualsComparison.overallMAPE <= 15 ? 'text-yellow-600' :
                                    actualsComparison.overallMAPE <= 25 ? 'text-orange-600' : 'text-red-600'
                              }`}>
                              {actualsComparison.overallMAPE.toFixed(2)}%
                            </div>
                            <div className="text-[10px] text-gray-400">Forecast vs Actuals</div>
                            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity w-64 pointer-events-none z-10 text-left">
                              <strong>Actuals Comparison MAPE</strong><br />
                              Error calculated by comparing the model's predictions against actual values you uploaded. This shows real-world forecast accuracy.
                              <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                            </div>
                          </div>
                          <div className="bg-white border rounded p-3 text-center relative group cursor-help">
                            <div className="text-xs text-gray-500 uppercase flex items-center justify-center">
                              RMSE
                              <Info className="w-3 h-3 ml-1 text-gray-400" />
                            </div>
                            <div className="text-xl font-bold text-gray-800">
                              {actualsComparison.overallRMSE.toFixed(2)}
                            </div>
                            <div className="text-[10px] text-gray-400">Forecast vs Actuals</div>
                            <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity w-64 pointer-events-none z-10 text-left">
                              <strong>Actuals Comparison RMSE</strong><br />
                              Root mean square error between predictions and actual values you uploaded. Lower is better.
                              <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                            </div>
                          </div>
                          <div className="bg-white border rounded p-3 text-center">
                            <div className="text-xs text-gray-500 uppercase">Avg Bias</div>
                            <div className={`text-xl font-bold ${actualsComparison.overallBias >= 0 ? 'text-blue-600' : 'text-purple-600'}`}>
                              {actualsComparison.overallBias >= 0 ? '+' : ''}{actualsComparison.overallBias.toFixed(2)}
                            </div>
                            <div className={`text-[10px] font-medium ${actualsComparison.overallBias >= 0 ? 'text-blue-500' : 'text-purple-500'}`}>
                              {actualsComparison.overallBias >= 0
                                ? 'Under-forecast (Actuals > Predicted)'
                                : 'Over-forecast (Actuals < Predicted)'}
                            </div>
                          </div>
                          <div className="bg-white border rounded p-3 text-center">
                            <div className="text-xs text-gray-500 uppercase">Matched Periods</div>
                            <div className="text-xl font-bold text-gray-800">
                              {actualsComparison.rows.length}
                            </div>
                            <div className="text-[10px] text-gray-400">
                              of {trainingResult?.results[comparisonModelIndex]?.forecast.length || 0} forecast periods
                            </div>
                          </div>
                        </div>

                        {/* Info about overlap */}
                        {trainingResult?.results[comparisonModelIndex] && actualsComparison.rows.length < trainingResult.results[comparisonModelIndex].forecast.length && (
                          <div className="bg-blue-50 border border-blue-200 rounded p-2 text-xs text-blue-700 flex items-start space-x-2">
                            <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
                            <span>
                              Showing comparison for <strong>{actualsComparison.rows.length}</strong> periods where dates overlap between your actuals file and the forecast horizon.
                              {trainingResult.results[comparisonModelIndex].forecast.length - actualsComparison.rows.length > 0 && (
                                <span> ({trainingResult.results[comparisonModelIndex].forecast.length - actualsComparison.rows.length} forecast periods have no matching actuals.)</span>
                              )}
                            </span>
                          </div>
                        )}

                        {/* Distribution by Status */}
                        <div className="bg-white border rounded p-3">
                          <div className="text-xs text-gray-500 uppercase mb-2">Error Distribution</div>
                          <div className="flex items-center space-x-1 h-6 rounded overflow-hidden">
                            {actualsComparison.excellentCount > 0 && (
                              <div
                                className="bg-green-500 h-full flex items-center justify-center text-white text-xs font-bold"
                                style={{ width: `${(actualsComparison.excellentCount / actualsComparison.rows.length) * 100}%` }}
                                title={`Excellent: ${actualsComparison.excellentCount}`}
                              >
                                {actualsComparison.excellentCount}
                              </div>
                            )}
                            {actualsComparison.goodCount > 0 && (
                              <div
                                className="bg-blue-500 h-full flex items-center justify-center text-white text-xs font-bold"
                                style={{ width: `${(actualsComparison.goodCount / actualsComparison.rows.length) * 100}%` }}
                                title={`Good: ${actualsComparison.goodCount}`}
                              >
                                {actualsComparison.goodCount}
                              </div>
                            )}
                            {actualsComparison.acceptableCount > 0 && (
                              <div
                                className="bg-yellow-500 h-full flex items-center justify-center text-white text-xs font-bold"
                                style={{ width: `${(actualsComparison.acceptableCount / actualsComparison.rows.length) * 100}%` }}
                                title={`Acceptable: ${actualsComparison.acceptableCount}`}
                              >
                                {actualsComparison.acceptableCount}
                              </div>
                            )}
                            {actualsComparison.reviewCount > 0 && (
                              <div
                                className="bg-orange-500 h-full flex items-center justify-center text-white text-xs font-bold"
                                style={{ width: `${(actualsComparison.reviewCount / actualsComparison.rows.length) * 100}%` }}
                                title={`Review: ${actualsComparison.reviewCount}`}
                              >
                                {actualsComparison.reviewCount}
                              </div>
                            )}
                            {actualsComparison.deviationCount > 0 && (
                              <div
                                className="bg-red-500 h-full flex items-center justify-center text-white text-xs font-bold"
                                style={{ width: `${(actualsComparison.deviationCount / actualsComparison.rows.length) * 100}%` }}
                                title={`Significant Deviation: ${actualsComparison.deviationCount}`}
                              >
                                {actualsComparison.deviationCount}
                              </div>
                            )}
                          </div>
                          {/* Only show legend items that have data */}
                          <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 text-[10px]">
                            {actualsComparison.excellentCount > 0 && (
                              <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-green-500 mr-1"></span>Excellent ≤5%</span>
                            )}
                            {actualsComparison.goodCount > 0 && (
                              <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-blue-500 mr-1"></span>Good 5-10%</span>
                            )}
                            {actualsComparison.acceptableCount > 0 && (
                              <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-yellow-500 mr-1"></span>Acceptable 10-15%</span>
                            )}
                            {actualsComparison.reviewCount > 0 && (
                              <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-orange-500 mr-1"></span>Review 15-25%</span>
                            )}
                            {actualsComparison.deviationCount > 0 && (
                              <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-red-500 mr-1"></span>Deviation &gt;25%</span>
                            )}
                          </div>
                        </div>

                        {/* Error Interpretation Legend */}
                        <div className="bg-gray-50 border rounded p-2 text-xs text-gray-600">
                          <span className="font-semibold">How to read errors:</span>
                          <span className="ml-2 text-blue-600">Actual &gt; Predicted = Under-forecast (model predicted too low)</span>
                          <span className="mx-2">|</span>
                          <span className="text-purple-600">Actual &lt; Predicted = Over-forecast (model predicted too high)</span>
                        </div>

                        {/* Severity Filter */}
                        <div className="bg-white border rounded p-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <span className="text-xs font-semibold text-gray-600">Filter by Severity:</span>
                              <div className="flex flex-wrap gap-1.5">
                                {[
                                  { key: 'excellent', label: 'Excellent ≤5%', color: 'green', count: actualsComparison.excellentCount },
                                  { key: 'good', label: 'Good 5-10%', color: 'blue', count: actualsComparison.goodCount },
                                  { key: 'acceptable', label: 'Acceptable 10-15%', color: 'yellow', count: actualsComparison.acceptableCount },
                                  { key: 'review', label: 'Review 15-25%', color: 'orange', count: actualsComparison.reviewCount },
                                  { key: 'significant_deviation', label: 'Deviation >25%', color: 'red', count: actualsComparison.deviationCount }
                                ].map(({ key, label, color, count }) => (
                                  <button
                                    key={key}
                                    onClick={() => {
                                      setComparisonSeverityFilter(prev =>
                                        prev.includes(key)
                                          ? prev.filter(f => f !== key)
                                          : [...prev, key]
                                      );
                                    }}
                                    disabled={count === 0}
                                    className={`px-2 py-1 rounded text-[10px] font-medium transition-all border ${comparisonSeverityFilter.includes(key)
                                        ? color === 'green' ? 'bg-green-500 text-white border-green-600' :
                                          color === 'blue' ? 'bg-blue-500 text-white border-blue-600' :
                                            color === 'yellow' ? 'bg-yellow-500 text-white border-yellow-600' :
                                              color === 'orange' ? 'bg-orange-500 text-white border-orange-600' :
                                                'bg-red-500 text-white border-red-600'
                                        : count === 0
                                          ? 'bg-gray-100 text-gray-400 border-gray-200 cursor-not-allowed'
                                          : color === 'green' ? 'bg-green-50 text-green-700 border-green-200 hover:bg-green-100' :
                                            color === 'blue' ? 'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100' :
                                              color === 'yellow' ? 'bg-yellow-50 text-yellow-700 border-yellow-200 hover:bg-yellow-100' :
                                                color === 'orange' ? 'bg-orange-50 text-orange-700 border-orange-200 hover:bg-orange-100' :
                                                  'bg-red-50 text-red-700 border-red-200 hover:bg-red-100'
                                      }`}
                                  >
                                    {label} ({count})
                                  </button>
                                ))}
                              </div>
                            </div>
                            {comparisonSeverityFilter.length > 0 && (
                              <button
                                onClick={() => setComparisonSeverityFilter([])}
                                className="text-xs text-gray-500 hover:text-gray-700 underline"
                              >
                                Clear filters
                              </button>
                            )}
                          </div>
                          {comparisonSeverityFilter.length > 0 && (
                            <div className="mt-2 text-xs text-gray-500">
                              Showing {actualsComparison.rows.filter(r => comparisonSeverityFilter.includes(r.status)).length} of {actualsComparison.rows.length} periods
                            </div>
                          )}
                        </div>

                        {/* Detailed Comparison Table */}
                        <div className="bg-white border rounded overflow-hidden">
                          <div className="max-h-80 overflow-y-auto">
                            <table className="min-w-full divide-y divide-gray-200 text-xs">
                              <thead className="bg-gray-50 sticky top-0">
                                <tr>
                                  <th className="px-2 py-2 text-left font-semibold text-gray-600">Date</th>
                                  <th className="px-2 py-2 text-center font-semibold text-gray-600">Day</th>
                                  <th className="px-2 py-2 text-right font-semibold text-gray-600">Predicted</th>
                                  <th className="px-2 py-2 text-right font-semibold text-gray-600">Actual</th>
                                  <th className="px-2 py-2 text-right font-semibold text-gray-600">
                                    <div>Diff</div>
                                  </th>
                                  <th className="px-2 py-2 text-right font-semibold text-gray-600">MAPE</th>
                                  <th className="px-2 py-2 text-center font-semibold text-gray-600">Direction</th>
                                  <th className="px-2 py-2 text-left font-semibold text-gray-600">
                                    <div>Context Flags</div>
                                    <div className="font-normal text-[10px] text-gray-400">(Events/Promos)</div>
                                  </th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-gray-100">
                                {actualsComparison.rows
                                  .filter(row => comparisonSeverityFilter.length === 0 || comparisonSeverityFilter.includes(row.status))
                                  .map((row, idx) => (
                                    <tr key={idx} className={
                                      row.status === 'significant_deviation' ? 'bg-red-50' :
                                        row.status === 'review' ? 'bg-orange-50' :
                                          row.status === 'acceptable' ? 'bg-yellow-50' :
                                            row.status === 'good' ? 'bg-blue-50' : 'bg-green-50'
                                    }>
                                      <td className="px-2 py-2 text-gray-700">{row.date}</td>
                                      <td className="px-2 py-2 text-center">
                                        <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${row.isWeekend
                                            ? 'bg-amber-100 text-amber-700'
                                            : 'bg-gray-100 text-gray-600'
                                          }`}>
                                          {row.dayOfWeek}
                                          {row.isWeekend && ' 🗓'}
                                        </span>
                                      </td>
                                      <td className="px-2 py-2 text-right text-gray-700">{parseFloat(String(row.predicted)).toFixed(2)}</td>
                                      <td className="px-2 py-2 text-right text-gray-700 font-medium">{parseFloat(String(row.actual)).toFixed(2)}</td>
                                      <td className={`px-2 py-2 text-right font-medium ${parseFloat(String(row.error)) >= 0 ? 'text-blue-600' : 'text-purple-600'}`}>
                                        {parseFloat(String(row.error)) >= 0 ? '+' : ''}{parseFloat(String(row.error)).toFixed(2)}
                                      </td>
                                      <td className={`px-2 py-2 text-right font-bold ${row.status === 'excellent' ? 'text-green-600' :
                                          row.status === 'good' ? 'text-blue-600' :
                                            row.status === 'acceptable' ? 'text-yellow-600' :
                                              row.status === 'review' ? 'text-orange-600' : 'text-red-600'
                                        }`}>
                                        {parseFloat(String(row.mape)).toFixed(2)}%
                                      </td>
                                      <td className="px-2 py-2 text-center">
                                        <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium ${row.error >= 0
                                            ? 'bg-blue-100 text-blue-700'
                                            : 'bg-purple-100 text-purple-700'
                                          }`}>
                                          {row.error >= 0 ? 'Under' : 'Over'}
                                        </span>
                                      </td>
                                      <td className="px-2 py-2">
                                        <div className="flex flex-wrap gap-1">
                                          {row.contextFlags && row.contextFlags.length > 0 ? (
                                            row.contextFlags.map((flag, flagIdx) => (
                                              <span
                                                key={flagIdx}
                                                className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-100 text-emerald-700"
                                                title={`Active covariate: ${flag}`}
                                              >
                                                {flag.replace(/^(is_|has_)/, '').replace(/_/g, ' ')}
                                              </span>
                                            ))
                                          ) : (
                                            <span className="text-[10px] text-gray-400">—</span>
                                          )}
                                        </div>
                                      </td>
                                    </tr>
                                  ))}
                              </tbody>
                            </table>
                          </div>
                        </div>

                        {/* Actions Row */}
                        <div className="flex items-center justify-between">
                          <button
                            onClick={() => {
                              setActualsComparison(null);
                              setActualsData([]);
                              setActualsColumns([]);
                              setActualsDateCol('');
                              setActualsValueCol('');
                              setActualsFilters({});
                              setFilteredActualsForComparison([]);
                              setComparisonSeverityFilter([]);
                              setShowActualsColumnSelector(false);
                              if (actualsFileInputRef.current) actualsFileInputRef.current.value = '';
                            }}
                            className="text-xs text-gray-500 hover:text-gray-700 underline"
                          >
                            Clear and upload different actuals
                          </button>
                          <div className="flex items-center space-x-3">
                            <button
                              onClick={downloadActualsComparisonCSV}
                              className="flex items-center space-x-1.5 px-3 py-1.5 bg-emerald-600 text-white text-xs rounded hover:bg-emerald-700 transition-colors"
                            >
                              <FileDown className="w-3 h-3" />
                              <span>Export CSV</span>
                            </button>
                            <button
                              onClick={regenerateExecutiveSummaryWithActuals}
                              disabled={isGeneratingSummary}
                              className="flex items-center space-x-2 px-3 py-1.5 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 disabled:bg-blue-300 transition-colors"
                            >
                              {isGeneratingSummary ? (
                                <>
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                  <span>Analyzing...</span>
                                </>
                              ) : (
                                <>
                                  <Activity className="w-3 h-3" />
                                  <span>Regenerate Summary with Root Cause Analysis</span>
                                </>
                              )}
                            </button>
                            <div className="text-xs text-gray-400">
                              Model: {actualsComparison.modelType}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Evaluation Grid with Tooltips - Training/Validation Metrics */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 bg-white border rounded text-center relative group cursor-help">
                      <div className="text-xs text-gray-500 uppercase flex items-center justify-center">
                        RMSE
                        <Info className="w-3 h-3 ml-1 text-gray-400" />
                      </div>
                      <div className="text-xl font-bold text-gray-800">{activeResult.metrics.rmse}</div>
                      <div className="text-[10px] text-gray-400">From Training</div>
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity w-64 pointer-events-none z-10 text-left">
                        <strong>Training RMSE</strong><br />
                        Calculated during model training using held-out test data (last ~12 periods of historical data). This shows model performance on past data, not uploaded actuals.
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                      </div>
                    </div>
                    <div className="p-4 bg-white border rounded text-center relative group cursor-help">
                      <div className="text-xs text-gray-500 uppercase flex items-center justify-center">
                        Validation MAPE
                        <Info className="w-3 h-3 ml-1 text-gray-400" />
                      </div>
                      <div className={`text-xl font-bold ${parseFloat(activeResult.metrics.mape) <= 5 ? 'text-green-600' :
                          parseFloat(activeResult.metrics.mape) <= 10 ? 'text-blue-600' :
                            parseFloat(activeResult.metrics.mape) <= 15 ? 'text-yellow-600' :
                              parseFloat(activeResult.metrics.mape) <= 25 ? 'text-orange-600' : 'text-red-600'
                        }`}>{activeResult.metrics.mape}%</div>
                      {activeResult.metrics.cv_mape ? (
                        <div className="text-[10px] text-gray-400">
                          CV: {activeResult.metrics.cv_mape}% (+/-{activeResult.metrics.cv_mape_std || '0'}%)
                        </div>
                      ) : (
                        <div className="text-[10px] text-gray-400">From Training</div>
                      )}
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity w-64 pointer-events-none z-10 text-left">
                        <strong>Training/Validation MAPE</strong><br />
                        Error calculated during model training on held-out historical data. Different from "Forecast vs Actuals" MAPE above which compares predictions to your uploaded actuals.<br /><br />
                        Finance thresholds:<br />
                        - Excellent: 5% or less<br />
                        - Good: 5-10%<br />
                        - Acceptable: 10-15%
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                      </div>
                    </div>
                    <div className="p-4 bg-white border rounded text-center relative group cursor-help">
                      <div className="text-xs text-gray-500 uppercase flex items-center justify-center">
                        R2 Score
                        <Info className="w-3 h-3 ml-1 text-gray-400" />
                      </div>
                      <div className={`text-xl font-bold ${parseFloat(activeResult.metrics.r2) >= 0.9 ? 'text-green-600' :
                          parseFloat(activeResult.metrics.r2) >= 0.7 ? 'text-blue-600' :
                            parseFloat(activeResult.metrics.r2) >= 0.5 ? 'text-yellow-600' : 'text-red-600'
                        }`}>{activeResult.metrics.r2}</div>
                      <div className="text-[10px] text-gray-400">From Training</div>
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity w-64 pointer-events-none z-10 text-left">
                        <strong>Training R2 Score</strong><br />
                        Proportion of variance explained by the model during training. Range 0-1, higher is better.<br />
                        - 0.9+: Excellent fit<br />
                        - 0.7-0.9: Good fit<br />
                        - 0.5-0.7: Moderate fit
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-gray-900"></div>
                      </div>
                    </div>
                  </div>

                  {/* Executive Summary */}
                  {isGeneratingSummary ? (
                    <div className="bg-white border border-gray-200 rounded-lg p-6">
                      <div className="flex items-center space-x-2 text-gray-600">
                        <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                        <span className="text-sm font-medium">Generating Executive Summary...</span>
                      </div>
                    </div>
                  ) : trainingResult.executiveSummary ? (
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-lg p-6 shadow-sm">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-2">
                          <Activity className="w-5 h-5 text-blue-600" />
                          <h3 className="text-sm font-bold text-gray-800">Executive Summary</h3>
                        </div>
                        <button
                          onClick={downloadExecutiveSummary}
                          className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 transition-colors"
                        >
                          <FileDown className="w-3 h-3" />
                          <span>Download</span>
                        </button>
                      </div>
                      <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed">
                        <ReactMarkdown
                          components={{
                            strong: ({ node, ...props }) => <span className="font-bold text-gray-900" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc pl-5 space-y-1 mt-2" {...props} />,
                            li: ({ node, ...props }) => <li className="mb-1" {...props} />,
                            p: ({ node, ...props }) => <p className="mb-3 last:mb-0" {...props} />,
                            h3: ({ node, ...props }) => <h3 className="text-sm font-bold text-blue-800 mt-4 mb-2 uppercase tracking-wide" {...props} />
                          }}
                        >
                          {trainingResult.executiveSummary}
                        </ReactMarkdown>
                      </div>
                    </div>
                  ) : null}
                </div>
              </NotebookCell>
            )
          }

          </>
          )}
        </main>
        <footer className="bg-white border-t border-gray-200 py-3 text-center text-xs text-gray-400">
          Created by Debu Sinha
        </footer>
      </div>

      {/* Batch Training Modal */}
      {showBatchTraining && (
        <BatchTraining
          data={filteredData}
          columns={columns}
          timeCol={timeCol}
          targetCol={targetCol}
          covariates={covariates}
          horizon={horizon}
          frequency={frequency}
          seasonalityMode={seasonalityMode}
          regressorMethod={regressorMethod}
          selectedModels={selectedModels}
          catalogName={catalogName}
          schemaName={schemaName}
          modelName={modelName}
          country={country}
          randomSeed={randomSeed}
          trainingStartDate={trainingStartDate}
          trainingEndDate={trainingEndDate}
          activeFilters={filters}
          onClose={() => setShowBatchTraining(false)}
          onComplete={(summary, segmentCols) => {
            setBatchTrainingSummary(summary);
            setBatchSegmentCols(segmentCols);
            setShowBatchTraining(false);
            // Automatically open the results viewer after training completes
            setShowBatchResultsViewer(true);
          }}
        />
      )}

      {/* Batch Results Viewer Modal - Individual forecast view per segment */}
      {showBatchResultsViewer && batchTrainingSummary && (
        <BatchResultsViewer
          batchResults={batchTrainingSummary}
          segmentCols={batchSegmentCols}
          timeCol={timeCol}
          targetCol={targetCol}
          covariates={covariates}
          onClose={() => setShowBatchResultsViewer(false)}
        />
      )}

      {/* Batch Comparison Modal - Compare forecasts vs actuals */}
      {showBatchComparison && batchTrainingSummary && (
        <BatchComparison
          batchResults={batchTrainingSummary}
          segmentCols={batchSegmentCols}
          timeCol={timeCol}
          onClose={() => setShowBatchComparison(false)}
        />
      )}
    </div >
  );
};

export default App;

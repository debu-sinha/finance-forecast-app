import React, { useState, useMemo, useRef } from 'react';
import {
  Target,
  Upload,
  FileDown,
  Filter,
  BarChart3,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import { DataRow, BatchTrainingSummary, BatchComparisonResult, BatchComparisonRow, MAPE_THRESHOLDS } from '../types';
import { parseCSV } from '../utils/csvParser';

interface BatchComparisonProps {
  batchResults: BatchTrainingSummary;
  segmentCols: string[];
  timeCol: string;
  onClose: () => void;
}

const getMapeStatus = (mape: number): 'excellent' | 'good' | 'acceptable' | 'review' | 'significant_deviation' => {
  if (mape <= MAPE_THRESHOLDS.EXCELLENT) return 'excellent';
  if (mape <= MAPE_THRESHOLDS.GOOD) return 'good';
  if (mape <= MAPE_THRESHOLDS.ACCEPTABLE) return 'acceptable';
  if (mape <= MAPE_THRESHOLDS.REVIEW) return 'review';
  return 'significant_deviation';
};

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'excellent': return 'bg-green-100 text-green-800 border-green-200';
    case 'good': return 'bg-blue-100 text-blue-800 border-blue-200';
    case 'acceptable': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    case 'review': return 'bg-orange-100 text-orange-800 border-orange-200';
    case 'significant_deviation': return 'bg-red-100 text-red-800 border-red-200';
    default: return 'bg-gray-100 text-gray-800 border-gray-200';
  }
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case 'excellent':
    case 'good':
      return <CheckCircle2 className="w-4 h-4 text-green-600" />;
    case 'acceptable':
      return <AlertTriangle className="w-4 h-4 text-yellow-600" />;
    case 'review':
      return <AlertTriangle className="w-4 h-4 text-orange-600" />;
    case 'significant_deviation':
      return <XCircle className="w-4 h-4 text-red-600" />;
    default:
      return null;
  }
};

export const BatchComparison: React.FC<BatchComparisonProps> = ({
  batchResults,
  segmentCols,
  timeCol,
  onClose
}) => {
  const [actualsData, setActualsData] = useState<DataRow[]>([]);
  const [actualsColumns, setActualsColumns] = useState<string[]>([]);
  const [actualsDateCol, setActualsDateCol] = useState<string>('');
  const [actualsValueCol, setActualsValueCol] = useState<string>('');
  const [comparisonResult, setComparisonResult] = useState<BatchComparisonResult | null>(null);
  const [filterStatus, setFilterStatus] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('📁 File upload triggered');
    const file = event.target.files?.[0];
    if (!file) {
      console.log('📁 No file selected');
      return;
    }
    console.log('📁 File selected:', file.name, file.size, 'bytes');

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      console.log('📁 File content loaded, length:', content?.length);
      const parsed = parseCSV(content);
      console.log('📁 Parsed rows:', parsed.length);

      if (parsed.length > 0) {
        setActualsData(parsed);
        setActualsColumns(Object.keys(parsed[0]));

        // Auto-detect date and value columns
        const cols = Object.keys(parsed[0]);
        const dateCol = cols.find(c =>
          c.toLowerCase().includes('date') ||
          c.toLowerCase().includes('ds') ||
          c.toLowerCase() === 'dt'
        );
        const valueCol = cols.find(c =>
          c.toLowerCase().includes('actual') ||
          c.toLowerCase().includes('value') ||
          c.toLowerCase() === 'y'
        );

        if (dateCol) setActualsDateCol(dateCol);
        if (valueCol) setActualsValueCol(valueCol);
      }
    };
    reader.readAsText(file);
  };

  // Parse flexible date format without timezone issues.
  // Adding 'T12:00:00' ensures we're at noon, avoiding midnight boundary issues
  // that can cause dates to shift when displayed in local timezone.
  const parseFlexibleDate = (dateStr: string): Date | null => {
    if (!dateStr) return null;

    // If it's already a full ISO string with time, use as-is
    if (dateStr.includes('T')) {
      const d = new Date(dateStr);
      if (!isNaN(d.getTime())) return d;
    }

    // For date-only strings, try various formats and add noon time
    // Check MM/DD/YY format (2-digit year) - common in DoorDash CSV data
    const mmddyyMatch = dateStr.match(/^(\d{1,2})\/(\d{1,2})\/(\d{2})$/);
    if (mmddyyMatch) {
      const [, month, day, yearStr] = mmddyyMatch;
      const year = 2000 + parseInt(yearStr);
      const d = new Date(year, parseInt(month) - 1, parseInt(day), 12, 0, 0);
      if (!isNaN(d.getTime())) return d;
    }

    // Check YYYY-MM-DD format
    const isoMatch = dateStr.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);
    if (isoMatch) {
      const d = new Date(dateStr + 'T12:00:00');
      if (!isNaN(d.getTime())) return d;
    }

    // Check MM/DD/YYYY format
    const usMatch = dateStr.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
    if (usMatch) {
      const [, month, day, year] = usMatch;
      const d = new Date(`${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}T12:00:00`);
      if (!isNaN(d.getTime())) return d;
    }

    // Try direct parsing as fallback but add noon time if it's a date-only string
    const d = new Date(dateStr + 'T12:00:00');
    if (!isNaN(d.getTime())) return d;

    // Last resort: direct parsing
    const fallback = new Date(dateStr);
    if (!isNaN(fallback.getTime())) return fallback;

    return null;
  };

  // Run comparison
  const handleCompare = () => {
    if (!actualsDateCol || !actualsValueCol || actualsData.length === 0) return;

    setIsProcessing(true);

    try {
      const comparisonRows: BatchComparisonRow[] = [];

      // Group actuals by segment
      const actualsMap = new Map<string, Map<string, number>>();

      actualsData.forEach(row => {
        // Build segment key
        const segmentParts: string[] = [];
        segmentCols.forEach(col => {
          if (col in row) {
            segmentParts.push(`${col}=${row[col]}`);
          }
        });
        const segmentKey = segmentParts.join(' | ');

        // Parse date
        const dateStr = String(row[actualsDateCol]);
        const d = parseFlexibleDate(dateStr);
        if (!d) return;

        const dateKey = d.toISOString().split('T')[0];

        if (!actualsMap.has(segmentKey)) {
          actualsMap.set(segmentKey, new Map());
        }

        // Sum values for duplicate dates - skip non-numeric values
        const segmentData = actualsMap.get(segmentKey)!;
        const rawVal = Number(row[actualsValueCol]);
        if (isNaN(rawVal)) return;
        const existingVal = segmentData.get(dateKey) || 0;
        segmentData.set(dateKey, existingVal + rawVal);
      });

      // Debug: Log all actuals segment keys and batch result segment keys
      console.log('🔍 Actuals segment keys:', Array.from(actualsMap.keys()));
      console.log('🔍 Batch result segment IDs:', batchResults.results.map(r => r.segmentId));
      console.log('🔍 Batch results structure:', batchResults.results.map(r => ({
        segmentId: r.segmentId,
        status: r.status,
        hasResult: !!r.result,
        modelCount: r.result?.results?.length || 0,
        hasForecast: r.result?.results?.some(m => m.forecast?.length > 0) || false
      })));

      // Match with forecast results and calculate real MAPE against actuals
      batchResults.results.forEach(result => {
        if (result.status === 'error') return;

        const segmentKey = result.segmentId;
        const segmentActuals = actualsMap.get(segmentKey);

        if (!segmentActuals || segmentActuals.size === 0) {
          // No actuals for this segment - still include with training metrics
          console.log(`❌ No actuals found for segment: "${segmentKey}"`);
          return;
        }

        // Get forecast data from the best model result
        const forecastMAPE = result.metrics?.mape || '0';
        const modelResults = result.result?.results || [];

        // Debug: Check what we received
        if (!result.result) {
          console.warn(`⚠️ Segment ${segmentKey}: result.result is undefined/null`);
        } else if (modelResults.length === 0) {
          console.warn(`⚠️ Segment ${segmentKey}: No models in result.result.results`);
        }

        // Find best model, or fall back to first successful model
        const bestModelResult = modelResults.find(r => r.isBest) ||
                                modelResults.find(r => r.forecast && r.forecast.length > 0) ||
                                modelResults[0];
        const forecastData = bestModelResult?.forecast || [];

        // Debug: Show warning if no forecast data available
        if (forecastData.length === 0) {
          console.warn(`⚠️ Segment ${segmentKey}: No forecast data. Models count: ${modelResults.length}, Model names: [${modelResults.map(m => m.modelName).join(', ')}], Best: ${bestModelResult?.modelName || 'none'}`);
        } else {
          // Show first forecast row to debug date column name
          const sampleRow = forecastData[0];
          console.log(`✓ Segment ${segmentKey}: ${forecastData.length} forecast rows. Columns: ${Object.keys(sampleRow).join(', ')}`);
        }

        // Build forecast map: date -> predicted value
        // Note: forecast data uses the user's time column name (e.g., 'Date'), not always 'ds'
        const forecastMap = new Map<string, number>();
        forecastData.forEach(row => {
          // Try the user's time column first, then common fallbacks
          const dateStr = String(row[timeCol] || row.ds || row.date || row.Date || '');
          const d = parseFlexibleDate(dateStr);
          if (d) {
            const dateKey = d.toISOString().split('T')[0];
            const yhat = Number(row.yhat ?? row.predicted ?? row.value ?? 0);
            forecastMap.set(dateKey, yhat);
          }
        });

        // Calculate real MAPE by comparing forecast vs actuals
        let totalAbsPercentError = 0;
        let totalError = 0; // For bias calculation
        let totalSquaredError = 0;
        let matchedPeriods = 0;

        segmentActuals.forEach((actualValue, dateKey) => {
          const forecastValue = forecastMap.get(dateKey);
          if (forecastValue !== undefined && actualValue !== 0) {
            const error = actualValue - forecastValue;
            const absPercentError = Math.abs(error / actualValue) * 100;
            totalAbsPercentError += absPercentError;
            totalError += error;
            totalSquaredError += error * error;
            matchedPeriods++;
          }
        });

        // Calculate metrics from actual comparison
        let actualMAPE: number;
        let actualBias: number;
        let actualRMSE: number;

        if (matchedPeriods > 0) {
          actualMAPE = totalAbsPercentError / matchedPeriods;
          actualBias = totalError / matchedPeriods;
          actualRMSE = Math.sqrt(totalSquaredError / matchedPeriods);
        } else {
          // No matching periods - fall back to training MAPE
          actualMAPE = parseFloat(forecastMAPE);
          actualBias = 0;
          actualRMSE = parseFloat(result.metrics?.rmse || '0');
        }

        comparisonRows.push({
          segmentId: segmentKey,
          filters: result.filters,
          forecastMAPE,
          actualMAPE,
          actualRMSE,
          actualBias,
          periodsCompared: matchedPeriods,
          status: getMapeStatus(actualMAPE),
          forecastRunId: result.runId
        });
      });

      // Calculate overall stats using period-weighted MAPE
      // Weights by number of matched forecast periods per segment, preventing
      // low-coverage segments from skewing the overall accuracy metric
      const totalAbsError = comparisonRows.reduce((sum, r) => sum + (r.actualMAPE * r.periodsCompared), 0);
      const totalPeriods = comparisonRows.reduce((sum, r) => sum + r.periodsCompared, 0);
      const overallMAPE = totalPeriods > 0 ? totalAbsError / totalPeriods : 0;

      const segmentsByStatus = {
        excellent: comparisonRows.filter(r => r.status === 'excellent').length,
        good: comparisonRows.filter(r => r.status === 'good').length,
        acceptable: comparisonRows.filter(r => r.status === 'acceptable').length,
        review: comparisonRows.filter(r => r.status === 'review').length,
        significant_deviation: comparisonRows.filter(r => r.status === 'significant_deviation').length
      };

      setComparisonResult({
        rows: comparisonRows,
        overallMAPE,
        totalSegments: comparisonRows.length,
        segmentsByStatus,
        comparisonDate: new Date().toISOString()
      });

    } catch (error) {
      console.error('Comparison failed:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Export scorecard to CSV
  const handleExportCSV = () => {
    if (!comparisonResult) return;

    const headers = [
      ...segmentCols,
      'training_mape',
      'actual_mape',
      'rmse',
      'bias',
      'periods_compared',
      'status',
      'run_id'
    ];

    const rows = comparisonResult.rows.map(r => {
      const segmentValues = segmentCols.map(col => r.filters[col] || '');
      return [
        ...segmentValues,
        r.forecastMAPE,
        r.actualMAPE.toFixed(2),
        r.actualRMSE.toFixed(2),
        r.actualBias.toFixed(2),
        r.periodsCompared,
        r.status,
        r.forecastRunId || ''
      ].map(v => `"${v}"`).join(',');
    });

    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_scorecard_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filtered results
  const filteredRows = useMemo(() => {
    if (!comparisonResult) return [];
    if (!filterStatus) return comparisonResult.rows;
    return comparisonResult.rows.filter(r => r.status === filterStatus);
  }, [comparisonResult, filterStatus]);

  // Diagnostic analysis of comparison results - Root Cause Analysis
  const diagnosticAnalysis = useMemo(() => {
    if (!comparisonResult || comparisonResult.rows.length === 0) return null;

    const rows = comparisonResult.rows;
    const issues: Array<{type: 'error' | 'warning' | 'info'; title: string; description: string; recommendation: string}> = [];

    // Calculate aggregate statistics
    const totalBias = rows.reduce((sum, r) => sum + r.actualBias, 0);
    const avgBias = rows.length > 0 ? totalBias / rows.length : 0;
    const biasStdDev = rows.length > 0 ? Math.sqrt(rows.reduce((sum, r) => sum + Math.pow(r.actualBias - avgBias, 2), 0) / rows.length) : 0;

    const positivesBias = rows.filter(r => r.actualBias > 0).length;
    const negativesBias = rows.filter(r => r.actualBias < 0).length;
    const totalBiasCount = positivesBias + negativesBias;
    const biasRatio = totalBiasCount > 0 ? positivesBias / totalBiasCount : 0.5;

    const highMapeSegments = rows.filter(r => r.actualMAPE > 20);
    const veryHighMapeSegments = rows.filter(r => r.actualMAPE > 30);
    const moderateMapeSegments = rows.filter(r => r.actualMAPE > 15 && r.actualMAPE <= 30);

    // Calculate MAPE std dev
    const mapeStdDev = rows.length > 0 ? Math.sqrt(rows.reduce((sum, r) => sum + Math.pow(r.actualMAPE - comparisonResult.overallMAPE, 2), 0) / rows.length) : 0;

    // Calculate training vs actual drift
    const trainActualDrifts = rows.map(r => r.actualMAPE - parseFloat(r.forecastMAPE || '0'));
    const avgDrift = trainActualDrifts.length > 0 ? trainActualDrifts.reduce((a, b) => a + b, 0) / trainActualDrifts.length : 0;

    // Check for no matched periods first - this is critical
    const noMatchSegments = rows.filter(r => r.periodsCompared === 0);
    const matchedSegments = rows.filter(r => r.periodsCompared > 0);

    // 0. Always show overall summary first
    const excellentGoodCount = comparisonResult.segmentsByStatus.excellent + comparisonResult.segmentsByStatus.good;
    const problemCount = comparisonResult.segmentsByStatus.review + comparisonResult.segmentsByStatus.significant_deviation;

    if (problemCount > 0 || noMatchSegments.length > 0) {
      issues.push({
        type: problemCount > rows.length * 0.3 ? 'error' : 'warning',
        title: 'Root Cause Analysis Summary',
        description: `Analyzed ${rows.length} segments: ${excellentGoodCount} performing well, ${comparisonResult.segmentsByStatus.acceptable} acceptable, ${problemCount} need attention${noMatchSegments.length > 0 ? `, ${noMatchSegments.length} have no data to compare` : ''}.`,
        recommendation: 'Review the specific issues below to understand what may be causing forecast deviations from actuals.'
      });
    }

    // 1. Check for no matched periods - often the biggest issue
    if (noMatchSegments.length > 0) {
      const percentage = ((noMatchSegments.length / rows.length) * 100).toFixed(0);
      issues.push({
        type: 'error',
        title: `${noMatchSegments.length} Segments (${percentage}%) Have No Overlapping Dates`,
        description: `Forecast dates don't align with actuals dates for: ${noMatchSegments.slice(0, 3).map(s => s.segmentId).join(', ')}${noMatchSegments.length > 3 ? ` and ${noMatchSegments.length - 3} more` : ''}.`,
        recommendation: 'Root causes: (1) Forecast horizon is different from actuals period - check date ranges, (2) Date format mismatch between forecast (ds column) and actuals file, (3) Weekly aggregation alignment - forecasts may be on different weekday than actuals, (4) Model training failed for these segments.'
      });
    }

    // Only continue detailed analysis if we have matched data
    if (matchedSegments.length > 0) {
      // 2. Check for systematic bias - most common root cause
      if (biasRatio > 0.65 && totalBiasCount >= 3) {
        issues.push({
          type: 'warning',
          title: 'Systematic Under-Forecasting Pattern',
          description: `${(biasRatio * 100).toFixed(0)}% of segments show actual > forecast. Average under-prediction: ${Math.abs(avgBias).toFixed(1)} units.`,
          recommendation: 'Root causes: (1) Recent demand growth not in training data - consider more recent training data, (2) Missing promotional/event features - actuals may include promotions not modeled, (3) Seasonality shift - holiday patterns may have changed, (4) New product launch or market expansion effects.'
        });
      } else if (biasRatio < 0.35 && totalBiasCount >= 3) {
        issues.push({
          type: 'warning',
          title: 'Systematic Over-Forecasting Pattern',
          description: `${((1 - biasRatio) * 100).toFixed(0)}% of segments show actual < forecast. Average over-prediction: ${Math.abs(avgBias).toFixed(1)} units.`,
          recommendation: 'Root causes: (1) Recent demand decline not captured - market may have shifted, (2) Supply constraints in actuals - stockouts reduce actual sales, (3) Competition or substitution effects, (4) Promotional period in training not repeated in forecast period.'
        });
      }

      // 3. Check for high variance
      if (mapeStdDev > 12 && rows.length >= 3) {
        issues.push({
          type: 'warning',
          title: 'Inconsistent Accuracy Across Segments',
          description: `MAPE ranges widely (std dev: ${mapeStdDev.toFixed(1)}%). Best: ${Math.min(...rows.map(r => r.actualMAPE)).toFixed(1)}%, Worst: ${Math.max(...rows.map(r => r.actualMAPE)).toFixed(1)}%.`,
          recommendation: 'Root causes: (1) Some segments have sparse or irregular data - check data volume per segment, (2) Different segments need different models - consider segment-specific training, (3) Data quality varies by segment - audit outliers in problem segments, (4) Some segments may have structural changes (new stores, products).'
        });
      }

      // 4. Check for segments with very high error
      if (veryHighMapeSegments.length > 0) {
        const worstSegments = veryHighMapeSegments
          .sort((a, b) => b.actualMAPE - a.actualMAPE)
          .slice(0, 3)
          .map(s => `${s.segmentId} (${s.actualMAPE.toFixed(1)}%)`);

        issues.push({
          type: 'error',
          title: `${veryHighMapeSegments.length} Segments with Major Deviations (>30% MAPE)`,
          description: `Problem segments: ${worstSegments.join(', ')}`,
          recommendation: 'Investigate these segments: (1) Check for data anomalies or outliers in training data, (2) Look for one-time events affecting actuals (promotions, stockouts), (3) Verify actuals data is correct for these segments, (4) Consider excluding from batch or modeling separately with different features.'
        });
      } else if (moderateMapeSegments.length > rows.length * 0.3) {
        issues.push({
          type: 'warning',
          title: `${moderateMapeSegments.length} Segments with Moderate Deviations (15-30% MAPE)`,
          description: `${((moderateMapeSegments.length / rows.length) * 100).toFixed(0)}% of segments have acceptable but improvable accuracy.`,
          recommendation: 'Consider: (1) Adding more features or regressors, (2) Tuning hyperparameters for these segments, (3) Using longer training history if available.'
        });
      }

      // 5. Check training vs actual MAPE drift
      const significantDriftSegments = rows.filter(r => Math.abs(r.actualMAPE - parseFloat(r.forecastMAPE || '0')) > 10);
      if (avgDrift > 8 && significantDriftSegments.length > 0) {
        issues.push({
          type: 'warning',
          title: 'Models Performing Worse on Actuals Than Training',
          description: `Actual MAPE is ${avgDrift.toFixed(1)}% higher than training MAPE on average. ${significantDriftSegments.length} segments show >10% degradation.`,
          recommendation: 'Root causes: (1) Overfitting - models learned training patterns too specifically, (2) Data drift - recent actuals have different patterns than training period, (3) Concept drift - relationships between features and target have changed, (4) Try simpler models or regularization.'
        });
      } else if (avgDrift < -8) {
        issues.push({
          type: 'info',
          title: 'Models Outperforming Training Expectations',
          description: `Actual MAPE is ${Math.abs(avgDrift).toFixed(1)}% better than training MAPE.`,
          recommendation: 'This is positive - actuals period may have more stable/predictable patterns. Training metrics may be conservative estimates.'
        });
      }

      // 6. Check for bias-dominated errors
      const highBiasSegments = matchedSegments.filter(r => r.actualRMSE > 0 && Math.abs(r.actualBias) > r.actualRMSE * 0.7);
      if (highBiasSegments.length > matchedSegments.length * 0.3 && highBiasSegments.length >= 2) {
        issues.push({
          type: 'info',
          title: 'Errors Are Primarily Systematic (Not Random)',
          description: `${highBiasSegments.length} segments (${((highBiasSegments.length / matchedSegments.length) * 100).toFixed(0)}%) have bias as the main error source.`,
          recommendation: 'Systematic errors are easier to fix than random errors: (1) Add trend components if missing, (2) Include level adjustment features, (3) Consider multiplicative seasonality mode, (4) Bias correction post-processing could help.'
        });
      }
    }

    // 7. Success message if no issues
    if (issues.length === 0) {
      issues.push({
        type: 'info',
        title: 'Forecast Performance is Strong',
        description: `Overall MAPE of ${comparisonResult.overallMAPE.toFixed(1)}% with ${excellentGoodCount} of ${rows.length} segments performing well.`,
        recommendation: 'Continue monitoring accuracy over time. Consider retraining periodically as more data becomes available.'
      });
    }

    return {
      issues,
      stats: {
        avgBias,
        biasStdDev,
        mapeStdDev,
        biasRatio,
        avgDrift,
        highMapeCount: highMapeSegments.length,
        noMatchCount: noMatchSegments.length,
        matchedCount: matchedSegments.length
      }
    };
  }, [comparisonResult]);

  const [showDiagnostics, setShowDiagnostics] = useState(true);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between bg-gradient-to-r from-emerald-50 to-teal-50">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-emerald-100 rounded-lg">
              <Target className="w-5 h-5 text-emerald-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Batch Comparison Scorecard</h2>
              <p className="text-sm text-gray-500">Compare batch forecasts against actuals</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
          >
            &times;
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Upload Actuals */}
          {!comparisonResult && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                <span className="bg-emerald-600 text-white w-5 h-5 rounded-full flex items-center justify-center text-xs mr-2">1</span>
                Upload Actuals Data
              </h3>
              <p className="text-xs text-gray-500 mb-3">
                Upload a CSV file with actuals data. Include segment columns ({segmentCols.join(', ')}) to match with forecasts.
              </p>

              <div className="flex items-center space-x-4">
                <label
                  className="flex items-center space-x-2 px-4 py-2 border-2 border-dashed border-gray-300 rounded-lg text-sm text-gray-600 hover:border-emerald-400 hover:text-emerald-600 transition-colors cursor-pointer bg-white"
                >
                  <Upload className="w-4 h-4" />
                  <span>Choose CSV File</span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </label>
                {actualsData.length === 0 && (
                  <span className="text-xs text-gray-400">No file selected</span>
                )}
              </div>

              {actualsData.length > 0 && (
                <div className="mt-4 p-3 bg-white rounded border border-gray-200">
                  <div className="text-xs text-gray-600 mb-2">
                    Loaded {actualsData.length} rows
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Date Column</label>
                      <select
                        value={actualsDateCol}
                        onChange={(e) => setActualsDateCol(e.target.value)}
                        className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm"
                      >
                        <option value="">Select...</option>
                        {actualsColumns.map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">Value Column</label>
                      <select
                        value={actualsValueCol}
                        onChange={(e) => setActualsValueCol(e.target.value)}
                        className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm"
                      >
                        <option value="">Select...</option>
                        {actualsColumns.map(col => (
                          <option key={col} value={col}>{col}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="mt-3 text-xs text-gray-500">
                    <strong>Segment columns detected:</strong>{' '}
                    {segmentCols.filter(c => actualsColumns.includes(c)).join(', ') || 'None'}
                    {segmentCols.some(c => !actualsColumns.includes(c)) && (
                      <span className="text-orange-600 ml-2">
                        (Missing: {segmentCols.filter(c => !actualsColumns.includes(c)).join(', ')})
                      </span>
                    )}
                  </div>

                  <button
                    onClick={handleCompare}
                    disabled={!actualsDateCol || !actualsValueCol || isProcessing}
                    className={`mt-3 flex items-center space-x-2 px-4 py-2 rounded text-sm font-medium text-white ${
                      !actualsDateCol || !actualsValueCol || isProcessing
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-emerald-600 hover:bg-emerald-700'
                    }`}
                  >
                    <BarChart3 className="w-4 h-4" />
                    <span>Generate Scorecard</span>
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Scorecard Results */}
          {comparisonResult && (
            <div className="space-y-4">
              {/* Warning if all segments have 0 periods compared */}
              {comparisonResult.rows.every(r => r.periodsCompared === 0) && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                  <div className="flex items-start space-x-3">
                    <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-sm font-semibold text-amber-800">No Forecast Data Available for Comparison</h4>
                      <p className="text-xs text-amber-700 mt-1">
                        The batch training results don't contain forecast data. This can happen if:
                      </p>
                      <ul className="text-xs text-amber-700 mt-1 ml-4 list-disc">
                        <li>All models failed during training (check MLflow for errors)</li>
                        <li>The forecast horizon extends beyond the actuals data dates</li>
                        <li>Date formats don't match between forecast and actuals</li>
                      </ul>
                      <p className="text-xs text-amber-700 mt-2">
                        The "Actual MAPE" shown below is using training MAPE as a fallback.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Summary Stats */}
              <div className="bg-gradient-to-r from-emerald-50 to-teal-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-gray-700 flex items-center">
                    <BarChart3 className="w-4 h-4 mr-2 text-emerald-600" />
                    Scorecard Summary
                  </h3>
                  <button
                    onClick={() => setShowDetails(!showDetails)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    {showDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>

                <div className="grid grid-cols-6 gap-3 text-center">
                  <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <div className="text-xl font-bold text-gray-800">{comparisonResult.totalSegments}</div>
                    <div className="text-xs text-gray-500">Segments</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-green-200">
                    <div className="text-xl font-bold text-green-600">{comparisonResult.segmentsByStatus.excellent}</div>
                    <div className="text-xs text-gray-500">Excellent</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-blue-200">
                    <div className="text-xl font-bold text-blue-600">{comparisonResult.segmentsByStatus.good}</div>
                    <div className="text-xs text-gray-500">Good</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-yellow-200">
                    <div className="text-xl font-bold text-yellow-600">{comparisonResult.segmentsByStatus.acceptable}</div>
                    <div className="text-xs text-gray-500">Acceptable</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-orange-200">
                    <div className="text-xl font-bold text-orange-600">{comparisonResult.segmentsByStatus.review}</div>
                    <div className="text-xs text-gray-500">Review</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-red-200">
                    <div className="text-xl font-bold text-red-600">{comparisonResult.segmentsByStatus.significant_deviation}</div>
                    <div className="text-xs text-gray-500">Deviation</div>
                  </div>
                </div>

                <div className="mt-3 text-center">
                  <span className="text-sm text-gray-600">
                    Overall MAPE (weighted): <strong className={`text-lg ${
                      comparisonResult.overallMAPE <= 10 ? 'text-green-600' :
                      comparisonResult.overallMAPE <= 15 ? 'text-yellow-600' : 'text-red-600'
                    }`}>{comparisonResult.overallMAPE.toFixed(2)}%</strong>
                  </span>
                </div>
              </div>

              {/* Root Cause Analysis Section */}
              {diagnosticAnalysis && (
                <div className="bg-white rounded-lg border border-gray-200 border-l-4 border-l-amber-400">
                  <div
                    className="px-4 py-3 border-b border-gray-100 flex items-center justify-between cursor-pointer hover:bg-gray-50 bg-amber-50"
                    onClick={() => setShowDiagnostics(!showDiagnostics)}
                  >
                    <h3 className="text-sm font-semibold text-gray-700 flex items-center">
                      <AlertTriangle className="w-4 h-4 mr-2 text-amber-600" />
                      Root Cause Analysis - Why Are Forecasts Deviating From Actuals?
                    </h3>
                    <button className="text-gray-500 hover:text-gray-700">
                      {showDiagnostics ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                  </div>

                  {showDiagnostics && (
                    <div className="p-4 space-y-3">
                      {/* Quick Stats */}
                      <div className="grid grid-cols-4 gap-2 mb-4">
                        <div className="bg-gray-50 rounded p-2 text-center">
                          <div className={`text-sm font-bold ${diagnosticAnalysis.stats.avgBias > 0 ? 'text-orange-600' : diagnosticAnalysis.stats.avgBias < 0 ? 'text-blue-600' : 'text-gray-600'}`}>
                            {diagnosticAnalysis.stats.avgBias > 0 ? '+' : ''}{diagnosticAnalysis.stats.avgBias.toFixed(1)}
                          </div>
                          <div className="text-[10px] text-gray-500">Avg Bias</div>
                        </div>
                        <div className="bg-gray-50 rounded p-2 text-center">
                          <div className="text-sm font-bold text-gray-700">
                            {(diagnosticAnalysis.stats.biasRatio * 100).toFixed(0)}%
                          </div>
                          <div className="text-[10px] text-gray-500">Under-forecast</div>
                        </div>
                        <div className="bg-gray-50 rounded p-2 text-center">
                          <div className="text-sm font-bold text-gray-700">
                            {diagnosticAnalysis.stats.mapeStdDev.toFixed(1)}%
                          </div>
                          <div className="text-[10px] text-gray-500">MAPE Std Dev</div>
                        </div>
                        <div className="bg-gray-50 rounded p-2 text-center">
                          <div className={`text-sm font-bold ${diagnosticAnalysis.stats.avgDrift > 0 ? 'text-red-600' : 'text-green-600'}`}>
                            {diagnosticAnalysis.stats.avgDrift > 0 ? '+' : ''}{diagnosticAnalysis.stats.avgDrift.toFixed(1)}%
                          </div>
                          <div className="text-[10px] text-gray-500">Train→Actual Drift</div>
                        </div>
                      </div>

                      {/* Issues List */}
                      {diagnosticAnalysis.issues.map((issue, idx) => (
                        <div
                          key={idx}
                          className={`p-3 rounded-lg border ${
                            issue.type === 'error' ? 'bg-red-50 border-red-200' :
                            issue.type === 'warning' ? 'bg-amber-50 border-amber-200' :
                            'bg-blue-50 border-blue-200'
                          }`}
                        >
                          <div className="flex items-start space-x-2">
                            {issue.type === 'error' ? (
                              <XCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                            ) : issue.type === 'warning' ? (
                              <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" />
                            ) : (
                              <CheckCircle2 className="w-4 h-4 text-blue-500 flex-shrink-0 mt-0.5" />
                            )}
                            <div className="flex-1">
                              <h4 className={`text-sm font-semibold ${
                                issue.type === 'error' ? 'text-red-800' :
                                issue.type === 'warning' ? 'text-amber-800' :
                                'text-blue-800'
                              }`}>
                                {issue.title}
                              </h4>
                              <p className={`text-xs mt-1 ${
                                issue.type === 'error' ? 'text-red-700' :
                                issue.type === 'warning' ? 'text-amber-700' :
                                'text-blue-700'
                              }`}>
                                {issue.description}
                              </p>
                              <div className={`mt-2 text-xs ${
                                issue.type === 'error' ? 'text-red-600' :
                                issue.type === 'warning' ? 'text-amber-600' :
                                'text-blue-600'
                              }`}>
                                <strong>Recommendation:</strong> {issue.recommendation}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Results Table */}
              {showDetails && (
                <div className="bg-white rounded-lg border border-gray-200">
                  {/* Filter Bar */}
                  <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Filter className="w-4 h-4 text-gray-400" />
                      <span className="text-xs text-gray-500">Filter:</span>
                      {['excellent', 'good', 'acceptable', 'review', 'significant_deviation'].map(status => {
                        const count = comparisonResult.segmentsByStatus[status as keyof typeof comparisonResult.segmentsByStatus];
                        if (count === 0) return null;

                        return (
                          <button
                            key={status}
                            onClick={() => setFilterStatus(filterStatus === status ? null : status)}
                            className={`px-2 py-1 rounded text-xs font-medium border transition-all ${
                              filterStatus === status
                                ? getStatusColor(status).replace(/bg-(\w+)-100/, 'bg-$1-500').replace(/text-(\w+)-800/, 'text-white').replace(/border-(\w+)-200/, 'border-$1-500')
                                : getStatusColor(status)
                            }`}
                          >
                            {status === 'significant_deviation' ? 'Deviation' : status.charAt(0).toUpperCase() + status.slice(1)} ({count})
                          </button>
                        );
                      })}
                      {filterStatus && (
                        <button
                          onClick={() => setFilterStatus(null)}
                          className="text-xs text-gray-500 hover:text-gray-700 underline"
                        >
                          Clear
                        </button>
                      )}
                    </div>
                    <span className="text-xs text-gray-400">
                      Showing {filteredRows.length} of {comparisonResult.rows.length}
                    </span>
                  </div>

                  {/* Table */}
                  <div className="max-h-60 overflow-y-auto">
                    <table className="min-w-full divide-y divide-gray-200 text-xs">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold text-gray-600">Segment</th>
                          <th className="px-3 py-2 text-center font-semibold text-gray-600">Status</th>
                          <th className="px-3 py-2 text-right font-semibold text-gray-600" title="MAPE from training/validation">Train MAPE</th>
                          <th className="px-3 py-2 text-right font-semibold text-gray-600" title="MAPE calculated from forecast vs uploaded actuals">Actual MAPE</th>
                          <th className="px-3 py-2 text-right font-semibold text-gray-600" title="Average forecast bias (positive = under-forecast)">Bias</th>
                          <th className="px-3 py-2 text-right font-semibold text-gray-600">RMSE</th>
                          <th className="px-3 py-2 text-right font-semibold text-gray-600">Periods</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {filteredRows.map((row, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            <td className="px-3 py-2 text-gray-700 font-medium">
                              {row.segmentId}
                            </td>
                            <td className="px-3 py-2 text-center">
                              <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${getStatusColor(row.status)}`}>
                                {getStatusIcon(row.status)}
                                <span className="ml-1">
                                  {row.status === 'significant_deviation' ? 'Deviation' : row.status.charAt(0).toUpperCase() + row.status.slice(1)}
                                </span>
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right text-gray-500">
                              {row.forecastMAPE}%
                            </td>
                            <td className="px-3 py-2 text-right font-medium">
                              <span className={row.periodsCompared === 0 ? 'text-gray-400 italic' : ''}>
                                {row.actualMAPE.toFixed(2)}%
                                {row.periodsCompared === 0 && <span className="text-xs ml-1" title="Using training MAPE - no forecast data available">*</span>}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right text-gray-600">
                              <span className={row.actualBias > 0 ? 'text-orange-600' : row.actualBias < 0 ? 'text-blue-600' : ''}>
                                {row.actualBias > 0 ? '+' : ''}{row.actualBias.toFixed(2)}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right text-gray-600">
                              {row.actualRMSE.toFixed(2)}
                            </td>
                            <td className="px-3 py-2 text-right">
                              <span className={row.periodsCompared === 0 ? 'text-amber-600 font-medium' : 'text-gray-600'}>
                                {row.periodsCompared === 0 ? '⚠ 0' : row.periodsCompared}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between bg-gray-50">
          <div className="text-xs text-gray-500">
            {comparisonResult && (
              <>Compared {comparisonResult.totalSegments} segments on {new Date(comparisonResult.comparisonDate).toLocaleDateString()}</>
            )}
          </div>
          <div className="flex items-center space-x-3">
            {comparisonResult && (
              <button
                onClick={handleExportCSV}
                className="flex items-center space-x-2 px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                <FileDown className="w-4 h-4" />
                <span>Export Scorecard</span>
              </button>
            )}
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

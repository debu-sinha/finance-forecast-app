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

        // Sum values for duplicate dates
        const segmentData = actualsMap.get(segmentKey)!;
        const existingVal = segmentData.get(dateKey) || 0;
        segmentData.set(dateKey, existingVal + Number(row[actualsValueCol]));
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

      // Calculate overall stats
      const mapes = comparisonRows.map(r => r.actualMAPE);
      const overallMAPE = mapes.length > 0 ? mapes.reduce((a, b) => a + b, 0) / mapes.length : 0;

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
                    Overall MAPE: <strong className={`text-lg ${
                      comparisonResult.overallMAPE <= 10 ? 'text-green-600' :
                      comparisonResult.overallMAPE <= 15 ? 'text-yellow-600' : 'text-red-600'
                    }`}>{comparisonResult.overallMAPE.toFixed(2)}%</strong>
                  </span>
                </div>
              </div>

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
                                ? getStatusColor(status).replace('100', '500').replace('800', 'white')
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

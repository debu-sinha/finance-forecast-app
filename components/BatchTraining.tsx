import React, { useState, useMemo } from 'react';
import {
  Layers,
  PlayCircle,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  FileDown,
  ChevronDown,
  ChevronUp,
  Filter,
  BarChart3,
  Settings2
} from 'lucide-react';
import { DataRow, BatchSegment, BatchTrainingSummary, BatchTrainingResult, MAPE_THRESHOLDS } from '../types';
import { trainBatchOnBackend, BatchTrainRequest, exportBatchResultsToCSV } from '../services/databricksApi';

interface BatchTrainingProps {
  data: DataRow[];
  columns: string[];
  timeCol: string;
  targetCol: string;
  covariates: string[];
  horizon: number;
  frequency: string;
  seasonalityMode: string;
  regressorMethod: string;
  selectedModels: string[];
  catalogName: string;
  schemaName: string;
  modelName: string;
  country: string;
  randomSeed: number;
  trainingStartDate?: string;
  trainingEndDate?: string;
  activeFilters?: Record<string, string>;
  onClose: () => void;
  onComplete?: (summary: BatchTrainingSummary, segmentCols: string[]) => void;
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

export const BatchTraining: React.FC<BatchTrainingProps> = ({
  data,
  columns,
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
  randomSeed,
  trainingStartDate,
  trainingEndDate,
  activeFilters,
  onClose,
  onComplete
}) => {
  // State for segment column selection
  const [selectedSegmentCols, setSelectedSegmentCols] = useState<string[]>([]);
  const [excludedSegments, setExcludedSegments] = useState<Set<string>>(new Set());
  const [isTraining, setIsTraining] = useState(false);
  const [trainingSummary, setTrainingSummary] = useState<BatchTrainingSummary | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<{ completed: number; total: number; current?: string }>({ completed: 0, total: 0 });
  const [showResults, setShowResults] = useState(true);
  const [filterStatus, setFilterStatus] = useState<string | null>(null);

  // Get available columns for segmentation (exclude time, target, and covariates)
  const availableSegmentCols = useMemo(() => {
    const excludeCols = new Set([timeCol, targetCol, ...covariates]);
    return columns.filter(col => !excludeCols.has(col));
  }, [columns, timeCol, targetCol, covariates]);

  // Calculate unique segments based on selected columns
  const segments = useMemo((): BatchSegment[] => {
    if (selectedSegmentCols.length === 0) return [];

    const segmentMap = new Map<string, BatchSegment>();

    data.forEach(row => {
      const filters: Record<string, string | number> = {};
      selectedSegmentCols.forEach(col => {
        filters[col] = row[col];
      });

      const id = selectedSegmentCols.map(col => `${col}=${row[col]}`).join(' | ');

      if (!segmentMap.has(id)) {
        segmentMap.set(id, {
          id,
          filters,
          data: [],
          rowCount: 0
        });
      }

      const segment = segmentMap.get(id)!;
      segment.data.push(row);
      segment.rowCount++;
    });

    return Array.from(segmentMap.values()).sort((a, b) => b.rowCount - a.rowCount);
  }, [data, selectedSegmentCols]);

  // Toggle segment column selection
  const toggleSegmentCol = (col: string) => {
    setSelectedSegmentCols(prev =>
      prev.includes(col)
        ? prev.filter(c => c !== col)
        : [...prev, col]
    );
    // Reset results and exclusions when changing columns
    setTrainingSummary(null);
    setExcludedSegments(new Set());
  };

  // Toggle segment exclusion
  const toggleSegmentExclusion = (segmentId: string) => {
    setExcludedSegments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(segmentId)) {
        newSet.delete(segmentId);
      } else {
        newSet.add(segmentId);
      }
      return newSet;
    });
  };

  // Get active segments (not excluded)
  const activeSegments = useMemo(() => {
    return segments.filter(s => !excludedSegments.has(s.id));
  }, [segments, excludedSegments]);

  // Start batch training
  const handleBatchTrain = async () => {
    if (activeSegments.length === 0) return;

    setIsTraining(true);
    setTrainingSummary(null);
    setTrainingProgress({ completed: 0, total: activeSegments.length });

    // Build requests for each active segment (excluded segments are skipped)
    const requests: BatchTrainRequest[] = activeSegments.map(segment => ({
      data: segment.data,
      timeCol,
      targetCol,
      covariates,
      horizon,
      frequency,
      seasonalityMode,
      regressorMethod,
      models: selectedModels,
      filters: segment.filters,
      catalogName,
      schemaName,
      modelName,
      country,
      randomSeed
    }));

    try {
      const summary = await trainBatchOnBackend(
        requests,
        2,
        (completed, total, latestResult) => {
          setTrainingProgress({
            completed,
            total,
            current: latestResult?.segmentId
          });
        }
      );

      setTrainingSummary(summary);
    } catch (error: any) {
      console.error('Batch training failed:', error);
    } finally {
      setIsTraining(false);
    }
  };

  // Export results to CSV
  const handleExportCSV = () => {
    if (!trainingSummary) return;

    const csv = exportBatchResultsToCSV(trainingSummary, selectedSegmentCols);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_forecast_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter results by status
  const filteredResults = useMemo(() => {
    if (!trainingSummary) return [];
    if (!filterStatus) return trainingSummary.results;

    return trainingSummary.results.filter(r => {
      if (filterStatus === 'error') return r.status === 'error';
      if (r.status === 'error') return false;
      const mape = parseFloat(r.metrics?.mape || '0');
      return getMapeStatus(mape) === filterStatus;
    });
  }, [trainingSummary, filterStatus]);

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between bg-gradient-to-r from-indigo-50 to-purple-50">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Layers className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Batch Training</h2>
              <p className="text-sm text-gray-500">Train forecasts for multiple segment combinations</p>
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
          {/* Configuration Summary - Shows what settings will be applied */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-blue-800 mb-3 flex items-center">
              <Settings2 className="w-4 h-4 mr-2" />
              Training Configuration (from main settings)
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div>
                <span className="text-blue-600 font-medium">Time Column:</span>
                <div className="text-gray-800 font-semibold">{timeCol}</div>
              </div>
              <div>
                <span className="text-blue-600 font-medium">Target Column:</span>
                <div className="text-gray-800 font-semibold">{targetCol}</div>
              </div>
              <div>
                <span className="text-blue-600 font-medium">Forecast Horizon:</span>
                <div className="text-gray-800 font-semibold">{horizon} periods</div>
              </div>
              <div>
                <span className="text-blue-600 font-medium">Frequency:</span>
                <div className="text-gray-800 font-semibold capitalize">{frequency}</div>
              </div>
              <div>
                <span className="text-blue-600 font-medium">Models:</span>
                <div className="text-gray-800 font-semibold">{selectedModels.join(', ')}</div>
              </div>
              <div>
                <span className="text-blue-600 font-medium">Seasonality:</span>
                <div className="text-gray-800 font-semibold capitalize">{seasonalityMode}</div>
              </div>
              <div>
                <span className="text-blue-600 font-medium">Covariates:</span>
                <div className="text-gray-800 font-semibold">{covariates.length > 0 ? covariates.join(', ') : 'None'}</div>
              </div>
              <div>
                <span className="text-blue-600 font-medium">Date Range:</span>
                <div className="text-gray-800 font-semibold">
                  {trainingStartDate || trainingEndDate
                    ? `${trainingStartDate || 'Start'} → ${trainingEndDate || 'End'}`
                    : 'All dates'}
                </div>
              </div>
              <div>
                <span className="text-blue-600 font-medium">Data Rows:</span>
                <div className="text-gray-800 font-semibold">{data.length.toLocaleString()} (after filters)</div>
              </div>
            </div>
            <p className="text-xs text-blue-600 mt-3 italic">
              ℹ️ These settings are inherited from the main configuration. Close this modal to modify them.
            </p>
            {activeFilters && Object.keys(activeFilters).length > 0 && (
              <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
                <strong>Category filters are active:</strong>{' '}
                {Object.entries(activeFilters).map(([k, v]) => `${k}="${v}"`).join(', ')}
                <br />
                <span className="text-red-600">
                  Batch training will only process this filtered subset. To train all segments,
                  close this modal and clear the filters in the main UI first.
                </span>
              </div>
            )}
            <p className="text-xs text-gray-500 mt-2">
              {trainingStartDate || trainingEndDate ? 'Date range filter has been applied. ' : ''}
              Batch training will split this data by the segment columns you select below — each unique combination gets its own forecast model.
            </p>
          </div>

          {/* Step 1: Select Segment Columns */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
              <span className="bg-indigo-600 text-white w-5 h-5 rounded-full flex items-center justify-center text-xs mr-2">1</span>
              Select Segment Columns
            </h3>
            <p className="text-xs text-gray-500 mb-3">
              Choose columns that define unique segments. Each unique combination will get its own forecast.
            </p>
            <div className="flex flex-wrap gap-2">
              {availableSegmentCols.map(col => (
                <button
                  key={col}
                  onClick={() => toggleSegmentCol(col)}
                  disabled={isTraining}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all border ${
                    selectedSegmentCols.includes(col)
                      ? 'bg-indigo-600 text-white border-indigo-600'
                      : 'bg-white text-gray-700 border-gray-300 hover:border-indigo-400'
                  } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {col}
                </button>
              ))}
            </div>
          </div>

          {/* Step 2: Preview Segments */}
          {selectedSegmentCols.length > 0 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
                <span className="bg-indigo-600 text-white w-5 h-5 rounded-full flex items-center justify-center text-xs mr-2">2</span>
                Segment Preview
                <span className="ml-2 px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded-full text-xs">
                  {activeSegments.length} of {segments.length} segments
                </span>
                {excludedSegments.size > 0 && (
                  <span className="ml-2 px-2 py-0.5 bg-orange-100 text-orange-700 rounded-full text-xs">
                    {excludedSegments.size} excluded
                  </span>
                )}
              </h3>
              <p className="text-xs text-gray-500 mb-2">
                Click on a row to exclude/include it from training.
              </p>
              <div className="max-h-48 overflow-y-auto bg-white rounded border border-gray-200">
                <table className="min-w-full divide-y divide-gray-200 text-xs">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="px-2 py-2 text-center font-semibold text-gray-600 w-8">✓</th>
                      {selectedSegmentCols.map(col => (
                        <th key={col} className="px-3 py-2 text-left font-semibold text-gray-600">
                          {col}
                        </th>
                      ))}
                      <th className="px-3 py-2 text-right font-semibold text-gray-600">Rows</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {segments.slice(0, 20).map((segment, idx) => {
                      const isExcluded = excludedSegments.has(segment.id);
                      return (
                        <tr
                          key={idx}
                          onClick={() => !isTraining && toggleSegmentExclusion(segment.id)}
                          className={`cursor-pointer transition-colors ${
                            isExcluded
                              ? 'bg-red-50 hover:bg-red-100 line-through opacity-60'
                              : 'hover:bg-gray-50'
                          } ${isTraining ? 'cursor-not-allowed' : ''}`}
                        >
                          <td className="px-2 py-2 text-center">
                            {isExcluded ? (
                              <XCircle className="w-4 h-4 text-red-500 inline" />
                            ) : (
                              <CheckCircle2 className="w-4 h-4 text-green-500 inline" />
                            )}
                          </td>
                          {selectedSegmentCols.map(col => (
                            <td key={col} className={`px-3 py-2 ${isExcluded ? 'text-gray-400' : 'text-gray-700'}`}>
                              {segment.filters[col]}
                            </td>
                          ))}
                          <td className={`px-3 py-2 text-right ${isExcluded ? 'text-gray-400' : 'text-gray-500'}`}>
                            {segment.rowCount}
                          </td>
                        </tr>
                      );
                    })}
                    {segments.length > 20 && (
                      <tr>
                        <td colSpan={selectedSegmentCols.length + 2} className="px-3 py-2 text-center text-gray-400 italic">
                          ... and {segments.length - 20} more segments
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Training Progress */}
          {isTraining && (
            <div className="bg-indigo-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-indigo-700">
                  Training Progress: {trainingProgress.completed} / {trainingProgress.total}
                </span>
                <Loader2 className="w-4 h-4 text-indigo-600 animate-spin" />
              </div>
              <div className="w-full bg-indigo-200 rounded-full h-2">
                <div
                  className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(trainingProgress.completed / trainingProgress.total) * 100}%` }}
                />
              </div>
              {trainingProgress.current && (
                <p className="text-xs text-indigo-600 mt-2">
                  Current: {trainingProgress.current}
                </p>
              )}
            </div>
          )}

          {/* Results */}
          {trainingSummary && (
            <div className="space-y-4">
              {/* Summary Stats */}
              <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-semibold text-gray-700 flex items-center">
                    <BarChart3 className="w-4 h-4 mr-2 text-green-600" />
                    Batch Training Complete
                  </h3>
                  <button
                    onClick={() => setShowResults(!showResults)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    {showResults ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>

                {/* MLflow Tracking Info */}
                {trainingSummary.batchId && (
                  <div className="mb-3 p-2 bg-blue-50 border border-blue-200 rounded text-xs">
                    <span className="font-medium text-blue-700">MLflow Batch ID:</span>{' '}
                    <code className="bg-blue-100 px-1.5 py-0.5 rounded text-blue-800">{trainingSummary.batchId}</code>
                    <span className="text-blue-600 ml-2">
                      — All {trainingSummary.totalSegments} segments are grouped in a single MLflow experiment for easy tracking
                    </span>
                    {/* Show link to first successful result's experiment */}
                    {trainingSummary.results.find(r => r.experimentUrl) && (
                      <a
                        href={trainingSummary.results.find(r => r.experimentUrl)?.experimentUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="ml-3 text-blue-700 underline hover:text-blue-900"
                      >
                        View in MLflow
                      </a>
                    )}
                  </div>
                )}

                <div className="grid grid-cols-4 gap-4 text-center">
                  <div className="bg-white rounded-lg p-3 border border-gray-200">
                    <div className="text-2xl font-bold text-gray-800">{trainingSummary.totalSegments}</div>
                    <div className="text-xs text-gray-500">Total Segments</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-green-200">
                    <div className="text-2xl font-bold text-green-600">{trainingSummary.successful}</div>
                    <div className="text-xs text-gray-500">Successful</div>
                  </div>
                  <div className="bg-white rounded-lg p-3 border border-red-200">
                    <div className="text-2xl font-bold text-red-600">{trainingSummary.failed}</div>
                    <div className="text-xs text-gray-500">Failed</div>
                  </div>
                  {trainingSummary.mapeStats && (
                    <div className="bg-white rounded-lg p-3 border border-blue-200">
                      <div className="text-2xl font-bold text-blue-600">{trainingSummary.mapeStats.mean.toFixed(1)}%</div>
                      <div className="text-xs text-gray-500">Avg MAPE</div>
                    </div>
                  )}
                </div>

                {trainingSummary.mapeStats && (
                  <div className="mt-3 flex items-center justify-center space-x-6 text-xs text-gray-600">
                    <span>Min: <strong>{trainingSummary.mapeStats.min.toFixed(1)}%</strong></span>
                    <span>Median: <strong>{trainingSummary.mapeStats.median.toFixed(1)}%</strong></span>
                    <span>Max: <strong>{trainingSummary.mapeStats.max.toFixed(1)}%</strong></span>
                  </div>
                )}
              </div>

              {/* Results Table */}
              {showResults && (
                <div className="bg-white rounded-lg border border-gray-200">
                  {/* Filter Bar */}
                  <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Filter className="w-4 h-4 text-gray-400" />
                      <span className="text-xs text-gray-500">Filter:</span>
                      {['excellent', 'good', 'acceptable', 'review', 'significant_deviation', 'error'].map(status => {
                        const count = trainingSummary.results.filter(r => {
                          if (status === 'error') return r.status === 'error';
                          if (r.status === 'error') return false;
                          const mape = parseFloat(r.metrics?.mape || '0');
                          return getMapeStatus(mape) === status;
                        }).length;

                        if (count === 0) return null;

                        return (
                          <button
                            key={status}
                            onClick={() => setFilterStatus(filterStatus === status ? null : status)}
                            className={`px-2 py-1 rounded text-xs font-medium border transition-all ${
                              filterStatus === status
                                ? status === 'error'
                                  ? 'bg-gray-800 text-white border-gray-800'
                                  : getStatusColor(status).replace('100', '500').replace('800', 'white')
                                : status === 'error'
                                  ? 'bg-gray-100 text-gray-600 border-gray-200'
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
                      Showing {filteredResults.length} of {trainingSummary.results.length}
                    </span>
                  </div>

                  {/* Table */}
                  <div className="max-h-60 overflow-y-auto">
                    <table className="min-w-full divide-y divide-gray-200 text-xs">
                      <thead className="bg-gray-50 sticky top-0">
                        <tr>
                          <th className="px-3 py-2 text-left font-semibold text-gray-600">Segment</th>
                          <th className="px-3 py-2 text-center font-semibold text-gray-600">Status</th>
                          <th className="px-3 py-2 text-left font-semibold text-gray-600">Best Model</th>
                          <th className="px-3 py-2 text-right font-semibold text-gray-600">MAPE</th>
                          <th className="px-3 py-2 text-right font-semibold text-gray-600">RMSE</th>
                          <th className="px-3 py-2 text-right font-semibold text-gray-600">CV MAPE</th>
                          <th className="px-3 py-2 text-center font-semibold text-gray-600">MLflow</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {filteredResults.map((result, idx) => {
                          const mape = result.metrics?.mape ? parseFloat(result.metrics.mape) : null;
                          const status = result.status === 'error' ? 'error' : (mape !== null ? getMapeStatus(mape) : 'unknown');

                          return (
                            <tr key={idx} className={result.status === 'error' ? 'bg-red-50' : 'hover:bg-gray-50'}>
                              <td className="px-3 py-2 text-gray-700 font-medium">
                                {result.segmentId}
                              </td>
                              <td className="px-3 py-2 text-center">
                                {result.status === 'success' ? (
                                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${getStatusColor(status)}`}>
                                    {status === 'significant_deviation' ? 'Deviation' : status.charAt(0).toUpperCase() + status.slice(1)}
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800 border border-red-200">
                                    <XCircle className="w-3 h-3 mr-1" />
                                    Error
                                  </span>
                                )}
                              </td>
                              <td className="px-3 py-2 text-gray-600">
                                {result.bestModel || '-'}
                              </td>
                              <td className="px-3 py-2 text-right font-medium">
                                {result.metrics?.mape ? `${result.metrics.mape}%` : '-'}
                              </td>
                              <td className="px-3 py-2 text-right text-gray-600">
                                {result.metrics?.rmse || '-'}
                              </td>
                              <td className="px-3 py-2 text-right text-gray-600">
                                {result.metrics?.cv_mape ? `${result.metrics.cv_mape}%` : '-'}
                              </td>
                              <td className="px-3 py-2 text-center">
                                {result.experimentUrl ? (
                                  <a
                                    href={result.experimentUrl}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-600 hover:text-blue-800 underline"
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    View
                                  </a>
                                ) : '-'}
                              </td>
                            </tr>
                          );
                        })}
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
            {selectedSegmentCols.length > 0 && !trainingSummary && (
              <>Models: {selectedModels.join(', ')} | Horizon: {horizon} {frequency}</>
            )}
            {trainingSummary && (
              <>Duration: {((trainingSummary.endTime! - trainingSummary.startTime) / 1000).toFixed(1)}s</>
            )}
          </div>
          <div className="flex items-center space-x-3">
            {trainingSummary && (
              <>
                <button
                  onClick={handleExportCSV}
                  className="flex items-center space-x-2 px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  <FileDown className="w-4 h-4" />
                  <span>Export CSV</span>
                </button>
                {onComplete && (
                  <button
                    onClick={() => onComplete(trainingSummary, selectedSegmentCols)}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium text-white"
                  >
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Done - Compare with Actuals</span>
                  </button>
                )}
              </>
            )}
            <button
              onClick={() => {
                if (trainingSummary && !window.confirm(
                  'You have unsaved batch training results. Are you sure you want to close?\n\n' +
                  'Click "Done - Compare with Actuals" to save results, or "Export CSV" to download.'
                )) {
                  return;
                }
                onClose();
              }}
              className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Close
            </button>
            {!trainingSummary && (
              <button
                onClick={handleBatchTrain}
                disabled={isTraining || activeSegments.length === 0}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium text-white ${
                  isTraining || activeSegments.length === 0
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700'
                }`}
              >
                {isTraining ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Training...</span>
                  </>
                ) : (
                  <>
                    <PlayCircle className="w-4 h-4" />
                    <span>Train {activeSegments.length} Segments</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

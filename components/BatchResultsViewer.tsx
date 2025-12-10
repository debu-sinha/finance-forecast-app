import React, { useState, useMemo } from 'react';
import {
  Layers,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ExternalLink,
  TrendingUp,
  BarChart3,
  Table2,
  LineChart,
  X,
  FileDown,
  Rocket,
  Loader2,
  Server
} from 'lucide-react';
import { BatchTrainingSummary, BatchTrainingResult, ModelRunResult, MAPE_THRESHOLDS, DataRow } from '../types';
import { ResultsChart } from './ResultsChart';
import { EvaluationChart } from './EvaluationChart';
import { ForecastTable } from './ForecastTable';
import { deployBatchModels } from '../services/databricksApi';

interface BatchResultsViewerProps {
  batchResults: BatchTrainingSummary;
  segmentCols: string[];
  timeCol: string;
  targetCol: string;
  covariates: string[];
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

export const BatchResultsViewer: React.FC<BatchResultsViewerProps> = ({
  batchResults,
  segmentCols,
  timeCol,
  targetCol,
  covariates,
  onClose
}) => {
  // Get successful results only
  const successfulResults = useMemo(() =>
    batchResults.results.filter(r => r.status === 'success' && r.result),
    [batchResults.results]
  );

  const [selectedSegmentIndex, setSelectedSegmentIndex] = useState(0);
  const [activeTab, setActiveTab] = useState<'overview' | 'chart' | 'table'>('overview');

  // Deploy modal state
  const [showDeployModal, setShowDeployModal] = useState(false);
  const [deployEndpointName, setDeployEndpointName] = useState('batch-forecast-endpoint');
  const [isDeploying, setIsDeploying] = useState(false);
  const [deployResult, setDeployResult] = useState<{status: 'success' | 'error', message: string, endpointUrl?: string} | null>(null);

  // Current selected segment
  const currentResult = successfulResults[selectedSegmentIndex];

  // Get the best model from current result
  const bestModel = useMemo(() => {
    if (!currentResult?.result?.results) return null;
    return currentResult.result.results.find(r => r.isBest) ||
           currentResult.result.results[0];
  }, [currentResult]);

  // Get history data from the result
  const historyData = useMemo(() => {
    return currentResult?.result?.history || [];
  }, [currentResult]);

  // Navigation
  const goToPrevious = () => {
    setSelectedSegmentIndex(prev => Math.max(0, prev - 1));
  };

  const goToNext = () => {
    setSelectedSegmentIndex(prev => Math.min(successfulResults.length - 1, prev + 1));
  };

  // Deploy all successful models as a router endpoint
  const handleDeploy = async () => {
    setIsDeploying(true);
    setDeployResult(null);

    try {
      const result = await deployBatchModels(
        batchResults,
        deployEndpointName,
        'main',
        'default',
        'finance_forecast_model'
      );

      if (result.status === 'success') {
        setDeployResult({
          status: 'success',
          message: result.message,
          endpointUrl: result.endpointUrl
        });
      } else {
        setDeployResult({
          status: 'error',
          message: result.message
        });
      }
    } catch (error: any) {
      setDeployResult({
        status: 'error',
        message: error.message || 'Deployment failed'
      });
    } finally {
      setIsDeploying(false);
    }
  };

  // Calculate MAPE status for current segment
  const currentMape = currentResult?.metrics?.mape ? parseFloat(currentResult.metrics.mape) : 0;
  const currentStatus = getMapeStatus(currentMape);

  if (successfulResults.length === 0) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg shadow-xl p-8 max-w-md text-center">
          <XCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No Successful Results</h3>
          <p className="text-gray-600 mb-4">All batch training segments failed. Check MLflow for error details.</p>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[95vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 bg-gradient-to-r from-purple-50 to-indigo-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Layers className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Batch Forecast Results</h2>
                <p className="text-sm text-gray-500">
                  {successfulResults.length} of {batchResults.totalSegments} segments successful
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Segment Navigator */}
        <div className="px-6 py-3 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center justify-between">
            {/* Segment Selector */}
            <div className="flex items-center space-x-3">
              <button
                onClick={goToPrevious}
                disabled={selectedSegmentIndex === 0}
                className={`p-2 rounded-lg border ${
                  selectedSegmentIndex === 0
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-white text-gray-700 hover:bg-gray-50 border-gray-300'
                }`}
              >
                <ChevronLeft className="w-5 h-5" />
              </button>

              <div className="flex items-center space-x-2">
                <select
                  value={selectedSegmentIndex}
                  onChange={(e) => setSelectedSegmentIndex(Number(e.target.value))}
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm font-medium bg-white min-w-[200px]"
                >
                  {successfulResults.map((result, idx) => (
                    <option key={idx} value={idx}>
                      {result.segmentId}
                    </option>
                  ))}
                </select>
                <span className="text-sm text-gray-500">
                  {selectedSegmentIndex + 1} of {successfulResults.length}
                </span>
              </div>

              <button
                onClick={goToNext}
                disabled={selectedSegmentIndex === successfulResults.length - 1}
                className={`p-2 rounded-lg border ${
                  selectedSegmentIndex === successfulResults.length - 1
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-white text-gray-700 hover:bg-gray-50 border-gray-300'
                }`}
              >
                <ChevronRight className="w-5 h-5" />
              </button>
            </div>

            {/* Status Badge */}
            <div className="flex items-center space-x-3">
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getStatusColor(currentStatus)}`}>
                {getStatusIcon(currentStatus)}
                <span className="ml-1.5">{currentStatus.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
              </span>
              {currentResult?.experimentUrl && (
                <a
                  href={currentResult.experimentUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-1 text-sm text-purple-600 hover:text-purple-700"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span>View in MLflow</span>
                </a>
              )}
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="px-6 py-2 border-b border-gray-200 bg-white">
          <div className="flex space-x-1">
            <button
              onClick={() => setActiveTab('overview')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'overview'
                  ? 'bg-purple-100 text-purple-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Overview</span>
            </button>
            <button
              onClick={() => setActiveTab('chart')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'chart'
                  ? 'bg-purple-100 text-purple-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <LineChart className="w-4 h-4" />
              <span>Forecast Chart</span>
            </button>
            <button
              onClick={() => setActiveTab('table')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'table'
                  ? 'bg-purple-100 text-purple-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Table2 className="w-4 h-4" />
              <span>Forecast Table</span>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'overview' && currentResult && bestModel && (
            <div className="space-y-6">
              {/* Metrics Summary */}
              <div className="grid grid-cols-4 gap-4">
                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="text-sm text-gray-500 mb-1">Best Model</div>
                  <div className="text-lg font-semibold text-gray-900">{bestModel.modelName}</div>
                </div>
                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="text-sm text-gray-500 mb-1">MAPE</div>
                  <div className={`text-lg font-semibold ${
                    currentMape <= 10 ? 'text-green-600' : currentMape <= 15 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {currentMape.toFixed(2)}%
                  </div>
                </div>
                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="text-sm text-gray-500 mb-1">RMSE</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {currentResult.metrics?.rmse || 'N/A'}
                  </div>
                </div>
                <div className="bg-white rounded-lg border border-gray-200 p-4">
                  <div className="text-sm text-gray-500 mb-1">R²</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {currentResult.metrics?.r2 || 'N/A'}
                  </div>
                </div>
              </div>

              {/* Model Cards */}
              {currentResult.result?.results && currentResult.result.results.length > 1 && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">All Trained Models</h3>
                  <div className="grid grid-cols-3 gap-3">
                    {currentResult.result.results.map((model, idx) => (
                      <div
                        key={idx}
                        className={`rounded-lg border p-3 ${
                          model.isBest
                            ? 'border-purple-300 bg-purple-50'
                            : 'border-gray-200 bg-white'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900 text-sm">{model.modelName}</span>
                          {model.isBest && (
                            <span className="px-2 py-0.5 bg-purple-600 text-white text-xs rounded-full">
                              BEST
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-3 gap-2 text-xs">
                          <div>
                            <span className="text-gray-500">MAPE</span>
                            <div className="font-medium">{model.metrics.mape}%</div>
                          </div>
                          <div>
                            <span className="text-gray-500">RMSE</span>
                            <div className="font-medium">{model.metrics.rmse}</div>
                          </div>
                          <div>
                            <span className="text-gray-500">R²</span>
                            <div className="font-medium">{model.metrics.r2}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Validation Chart */}
              {bestModel.validation && bestModel.validation.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Validation Performance</h3>
                  <div className="bg-white rounded-lg border border-gray-200 p-4">
                    <EvaluationChart
                      validation={bestModel.validation}
                      timeCol={timeCol}
                      targetCol={targetCol}
                    />
                  </div>
                </div>
              )}

              {/* Quick Forecast Preview */}
              {bestModel.forecast && bestModel.forecast.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">
                    Forecast Preview ({bestModel.forecast.length} periods)
                  </h3>
                  <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left font-semibold text-gray-600">Period</th>
                          <th className="px-4 py-2 text-right font-semibold text-gray-600">Forecast</th>
                          <th className="px-4 py-2 text-right font-semibold text-gray-600">Lower 95%</th>
                          <th className="px-4 py-2 text-right font-semibold text-gray-600">Upper 95%</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {bestModel.forecast.slice(0, 5).map((row, idx) => (
                          <tr key={idx} className="hover:bg-gray-50">
                            <td className="px-4 py-2 text-gray-700">
                              {row[timeCol] || row.ds || row.date}
                            </td>
                            <td className="px-4 py-2 text-right font-medium text-gray-900">
                              {typeof row.yhat === 'number' ? row.yhat.toLocaleString(undefined, {maximumFractionDigits: 2}) : row.yhat}
                            </td>
                            <td className="px-4 py-2 text-right text-gray-500">
                              {typeof row.yhat_lower === 'number' ? row.yhat_lower.toLocaleString(undefined, {maximumFractionDigits: 2}) : row.yhat_lower}
                            </td>
                            <td className="px-4 py-2 text-right text-gray-500">
                              {typeof row.yhat_upper === 'number' ? row.yhat_upper.toLocaleString(undefined, {maximumFractionDigits: 2}) : row.yhat_upper}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {bestModel.forecast.length > 5 && (
                      <div className="px-4 py-2 bg-gray-50 text-sm text-gray-500 text-center">
                        +{bestModel.forecast.length - 5} more periods.
                        <button
                          onClick={() => setActiveTab('table')}
                          className="text-purple-600 hover:underline ml-1"
                        >
                          View all
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'chart' && currentResult && bestModel && (
            <div className="space-y-4">
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                {bestModel.forecast && bestModel.forecast.length > 0 ? (
                  <ResultsChart
                    history={historyData}
                    validation={bestModel.validation || []}
                    forecast={bestModel.forecast}
                    timeCol={timeCol}
                    targetCol={targetCol}
                    covariates={covariates}
                  />
                ) : (
                  <div className="text-center py-12 text-gray-500">
                    <LineChart className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                    <p>No forecast data available for this segment</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'table' && currentResult && bestModel && (
            <div className="space-y-4">
              {bestModel.forecast && bestModel.forecast.length > 0 ? (
                <ForecastTable
                  forecast={bestModel.forecast}
                  timeCol={timeCol}
                  targetCol={targetCol}
                />
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <Table2 className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                  <p>No forecast data available for this segment</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex items-center justify-between">
          <div className="text-sm text-gray-500">
            Segment: <span className="font-medium text-gray-700">{currentResult?.segmentId}</span>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowDeployModal(true)}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 flex items-center space-x-2"
            >
              <Rocket className="w-4 h-4" />
              <span>Deploy All ({successfulResults.length})</span>
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Close
            </button>
          </div>
        </div>
      </div>

      {/* Deploy Modal */}
      {showDeployModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[60]">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6">
            <div className="flex items-center space-x-3 mb-4">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Server className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Deploy Batch Models</h3>
                <p className="text-sm text-gray-500">
                  Deploy all {successfulResults.length} segment models as a single router endpoint
                </p>
              </div>
            </div>

            {!deployResult && (
              <>
                <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-start space-x-2">
                    <Server className="w-4 h-4 text-blue-600 mt-0.5" />
                    <div className="text-xs text-blue-700">
                      <strong>Router Endpoint Approach:</strong> A single endpoint will be created that routes requests
                      to the appropriate segment model based on the input filters. This is more cost-effective than
                      deploying separate endpoints per segment.
                    </div>
                  </div>
                </div>

                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Endpoint Name
                  </label>
                  <input
                    type="text"
                    value={deployEndpointName}
                    onChange={(e) => setDeployEndpointName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    placeholder="batch-forecast-endpoint"
                  />
                </div>

                <div className="mb-4 text-xs text-gray-500">
                  <strong>Segments to deploy:</strong>
                  <div className="mt-1 max-h-24 overflow-y-auto">
                    {successfulResults.map((r, i) => (
                      <div key={i} className="flex items-center space-x-2 py-0.5">
                        <CheckCircle2 className="w-3 h-3 text-green-500" />
                        <span>{r.segmentId}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {deployResult && (
              <div className={`mb-4 p-4 rounded-lg ${
                deployResult.status === 'success'
                  ? 'bg-green-50 border border-green-200'
                  : 'bg-red-50 border border-red-200'
              }`}>
                <div className="flex items-center space-x-2">
                  {deployResult.status === 'success' ? (
                    <CheckCircle2 className="w-5 h-5 text-green-600" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-600" />
                  )}
                  <span className={`font-medium ${
                    deployResult.status === 'success' ? 'text-green-700' : 'text-red-700'
                  }`}>
                    {deployResult.message}
                  </span>
                </div>
                {deployResult.endpointUrl && (
                  <a
                    href={deployResult.endpointUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-2 inline-flex items-center space-x-1 text-sm text-purple-600 hover:text-purple-700"
                  >
                    <ExternalLink className="w-4 h-4" />
                    <span>View Endpoint</span>
                  </a>
                )}
              </div>
            )}

            <div className="flex items-center justify-end space-x-3">
              <button
                onClick={() => {
                  setShowDeployModal(false);
                  setDeployResult(null);
                }}
                className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                {deployResult ? 'Close' : 'Cancel'}
              </button>
              {!deployResult && (
                <button
                  onClick={handleDeploy}
                  disabled={isDeploying || !deployEndpointName}
                  className={`px-4 py-2 rounded-lg text-sm font-medium text-white flex items-center space-x-2 ${
                    isDeploying || !deployEndpointName
                      ? 'bg-gray-400 cursor-not-allowed'
                      : 'bg-purple-600 hover:bg-purple-700'
                  }`}
                >
                  {isDeploying ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Deploying...</span>
                    </>
                  ) : (
                    <>
                      <Rocket className="w-4 h-4" />
                      <span>Deploy</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

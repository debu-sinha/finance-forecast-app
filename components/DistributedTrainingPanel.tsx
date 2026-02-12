/**
 * DistributedTrainingPanel - Handles both local and cluster-based training.
 *
 * This component provides a unified interface that:
 * - Works locally for testing (uses legacy training in app container)
 * - Works with Databricks cluster when deployed as an App (uses job delegation)
 *
 * It auto-detects whether cluster delegation is available and adjusts the UI accordingly.
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Server,
  Laptop,
  Cloud,
  PlayCircle,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Settings2,
  Info,
  RefreshCw,
} from 'lucide-react';
import { DataRow, TrainingMode, TrainingJob, JobConfig, ForecastResult } from '../types';
import { logger } from '../utils/logger';
import { TrainingModeSelector } from './TrainingModeSelector';
import { JobStatusPanel } from './JobStatusPanel';
import {
  getDelegationStatus,
  createJob,
  submitJob,
  getJobResults,
  DelegationStatus,
} from '../services/jobApi';
import { trainModelOnBackend } from '../services/databricksApi';

interface DistributedTrainingPanelProps {
  // Data configuration
  data: DataRow[];
  timeCol: string;
  targetCol: string;
  covariates: string[];
  horizon: number;
  frequency: string;
  seasonalityMode: string;
  selectedModels: string[];

  // Optional configuration
  idCol?: string;
  catalogName?: string;
  schemaName?: string;
  modelName?: string;
  country?: string;
  randomSeed?: number;

  // Callbacks
  onTrainingComplete?: (result: ForecastResult) => void;
  onTrainingError?: (error: Error) => void;
}

type TrainingEnvironment = 'local' | 'cluster' | 'auto';

export const DistributedTrainingPanel: React.FC<DistributedTrainingPanelProps> = ({
  data,
  timeCol,
  targetCol,
  covariates,
  horizon,
  frequency,
  seasonalityMode,
  selectedModels,
  idCol,
  catalogName = 'main',
  schemaName = 'default',
  modelName = 'finance_forecast_model',
  country = 'US',
  randomSeed = 42,
  onTrainingComplete,
  onTrainingError,
}) => {
  // State
  const [delegationStatus, setDelegationStatus] = useState<DelegationStatus | null>(null);
  const [checkingDelegation, setCheckingDelegation] = useState(true);
  const [trainingEnvironment, setTrainingEnvironment] = useState<TrainingEnvironment>('auto');
  const [trainingMode, setTrainingMode] = useState<TrainingMode>('autogluon');
  const [isTraining, setIsTraining] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [localProgress, setLocalProgress] = useState<string>('');

  // Check delegation status on mount
  useEffect(() => {
    checkDelegationStatus();
  }, []);

  const checkDelegationStatus = async () => {
    setCheckingDelegation(true);
    try {
      const status = await getDelegationStatus();
      setDelegationStatus(status);

      // Auto-select environment based on availability
      if (status.enabled) {
        setTrainingEnvironment('cluster');
      } else {
        setTrainingEnvironment('local');
        setTrainingMode('legacy');
      }
    } catch (err) {
      logger.error('Failed to check delegation status:', err);
      setDelegationStatus({ enabled: false, message: 'Failed to check cluster availability' });
      setTrainingEnvironment('local');
      setTrainingMode('legacy');
    } finally {
      setCheckingDelegation(false);
    }
  };

  // Determine effective environment
  const effectiveEnvironment = trainingEnvironment === 'auto'
    ? (delegationStatus?.enabled ? 'cluster' : 'local')
    : trainingEnvironment;

  // Start training
  const handleStartTraining = async () => {
    setError(null);
    setIsTraining(true);

    try {
      if (effectiveEnvironment === 'cluster') {
        await startClusterTraining();
      } else {
        await startLocalTraining();
      }
    } catch (err) {
      logger.error('Training failed:', err);
      setError(err instanceof Error ? err.message : 'Training failed');
      if (onTrainingError) {
        onTrainingError(err instanceof Error ? err : new Error('Training failed'));
      }
      setIsTraining(false);
    }
  };

  // Cluster-based training
  const startClusterTraining = async () => {
    const jobConfig: JobConfig = {
      data,
      time_col: timeCol,
      target_col: targetCol,
      id_col: idCol,
      covariates,
      horizon,
      frequency,
      training_mode: trainingMode,
      models: selectedModels,
      seasonality_mode: seasonalityMode,
      time_limit: 600,
      presets: 'medium_quality',
    };

    // Create the job
    setLocalProgress('Creating job...');
    const job = await createJob(jobConfig);

    // Submit immediately
    setLocalProgress('Submitting to cluster...');
    const submittedJob = await submitJob(job.job_id);
    setCurrentJobId(submittedJob.job_id);
  };

  // Local training (legacy mode)
  const startLocalTraining = async () => {
    setLocalProgress('Starting local training...');

    const result = await trainModelOnBackend(
      data,
      timeCol,
      targetCol,
      covariates,
      horizon,
      frequency,
      seasonalityMode,
      'mean',
      selectedModels,
      catalogName,
      schemaName,
      modelName,
      country,
      undefined, // filters
      undefined, // fromDate
      undefined, // toDate
      randomSeed
    );

    // Transform result to ForecastResult format
    const forecastResult: ForecastResult = {
      history: result.history || [],
      results: result.models?.map((m: any) => ({
        modelType: m.model_type,
        modelName: m.model_name,
        isBest: m.is_best,
        metrics: m.metrics,
        hyperparameters: m.hyperparameters || {},
        validation: m.validation || [],
        forecast: m.forecast || [],
        experimentUrl: m.experiment_url,
        runUrl: m.run_url,
      })) || [],
      explanation: result.explanation || '',
      pythonCode: result.python_code || '',
    };

    setIsTraining(false);
    setLocalProgress('');

    if (onTrainingComplete) {
      onTrainingComplete(forecastResult);
    }
  };

  // Handle cluster job completion
  const handleJobComplete = useCallback(async (job: TrainingJob) => {
    if (job.status === 'completed' && job.results) {
      // Fetch full results
      try {
        const results = await getJobResults(job.job_id);

        // Transform to ForecastResult
        const forecastResult: ForecastResult = {
          history: [],
          results: results.leaderboard?.map((item: any) => ({
            modelType: item.model?.toLowerCase() || 'unknown',
            modelName: item.model || 'Unknown',
            isBest: item.model === results.best_model,
            metrics: {
              mape: String(results.mape || 0),
              rmse: '0',
              r2: '0',
            },
            hyperparameters: {},
            validation: [],
            forecast: results.forecast || [],
          })) || [],
          explanation: `Best model: ${results.best_model} with MAPE: ${results.mape?.toFixed(2)}%`,
          pythonCode: '',
        };

        if (onTrainingComplete) {
          onTrainingComplete(forecastResult);
        }
      } catch (err) {
        logger.error('Failed to fetch job results:', err);
        setError('Training completed but failed to fetch results');
      }
    } else if (job.status === 'failed') {
      setError(job.error || 'Training failed');
      if (onTrainingError) {
        onTrainingError(new Error(job.error || 'Training failed'));
      }
    }

    setIsTraining(false);
    setCurrentJobId(null);
    setLocalProgress('');
  }, [onTrainingComplete, onTrainingError]);

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Server className="w-5 h-5 text-indigo-600" />
            <h3 className="text-sm font-semibold text-gray-800">Training Environment</h3>
          </div>
          <button
            onClick={checkDelegationStatus}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-white rounded transition-colors"
            title="Refresh cluster status"
          >
            <RefreshCw className={`w-4 h-4 ${checkingDelegation ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Environment selector */}
        {checkingDelegation ? (
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Checking cluster availability...</span>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Environment toggle */}
            <div className="flex items-center space-x-2">
              <span className="text-xs font-medium text-gray-600">Run on:</span>
              <div className="flex rounded-lg border border-gray-300 overflow-hidden">
                <button
                  onClick={() => setTrainingEnvironment('local')}
                  disabled={isTraining}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium transition-colors ${
                    effectiveEnvironment === 'local'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white text-gray-700 hover:bg-gray-50'
                  } ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <Laptop className="w-3.5 h-3.5" />
                  <span>Local</span>
                </button>
                <button
                  onClick={() => setTrainingEnvironment('cluster')}
                  disabled={isTraining || !delegationStatus?.enabled}
                  className={`flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium border-l border-gray-300 transition-colors ${
                    effectiveEnvironment === 'cluster'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-white text-gray-700 hover:bg-gray-50'
                  } ${(isTraining || !delegationStatus?.enabled) ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <Cloud className="w-3.5 h-3.5" />
                  <span>Cluster</span>
                </button>
              </div>

              {/* Status indicator */}
              {delegationStatus?.enabled ? (
                <span className="flex items-center text-xs text-green-600">
                  <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                  Cluster available
                </span>
              ) : (
                <span className="flex items-center text-xs text-yellow-600">
                  <AlertTriangle className="w-3.5 h-3.5 mr-1" />
                  Local only
                </span>
              )}
            </div>

            {/* Training mode selector (only for cluster) */}
            {effectiveEnvironment === 'cluster' && delegationStatus?.enabled && (
              <TrainingModeSelector
                selectedMode={trainingMode}
                onModeChange={setTrainingMode}
                disabled={isTraining}
              />
            )}

            {/* Info box */}
            <div className={`flex items-start space-x-2 p-3 rounded-lg text-xs ${
              effectiveEnvironment === 'cluster'
                ? 'bg-indigo-50 text-indigo-700'
                : 'bg-gray-50 text-gray-600'
            }`}>
              <Info className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <div>
                {effectiveEnvironment === 'cluster' ? (
                  <>
                    <strong>Cluster Training:</strong> Your training job will run on a dedicated
                    Databricks cluster ({delegationStatus?.cluster_id?.substring(0, 12)}...).
                    This supports AutoGluon, StatsForecast, NeuralForecast, and Many Model Forecasting.
                    Results are tracked in MLflow.
                  </>
                ) : (
                  <>
                    <strong>Local Training:</strong> Training runs in the app container using
                    Prophet/ARIMA models. Good for testing and small datasets. For production
                    workloads, deploy as a Databricks App with cluster delegation enabled.
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Configuration summary */}
        <div className="bg-gray-50 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Settings2 className="w-4 h-4 text-gray-500" />
            <span className="text-xs font-medium text-gray-700">Configuration</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Time:</span>{' '}
              <span className="font-medium text-gray-800">{timeCol}</span>
            </div>
            <div>
              <span className="text-gray-500">Target:</span>{' '}
              <span className="font-medium text-gray-800">{targetCol}</span>
            </div>
            <div>
              <span className="text-gray-500">Horizon:</span>{' '}
              <span className="font-medium text-gray-800">{horizon} periods</span>
            </div>
            <div>
              <span className="text-gray-500">Frequency:</span>{' '}
              <span className="font-medium text-gray-800">{frequency}</span>
            </div>
            <div>
              <span className="text-gray-500">Models:</span>{' '}
              <span className="font-medium text-gray-800">
                {effectiveEnvironment === 'cluster' ? trainingMode : selectedModels.join(', ')}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Data:</span>{' '}
              <span className="font-medium text-gray-800">{data.length.toLocaleString()} rows</span>
            </div>
          </div>
        </div>

        {/* Job status panel (for cluster jobs) */}
        {currentJobId && (
          <JobStatusPanel
            jobId={currentJobId}
            onComplete={handleJobComplete}
            onCancel={() => {
              setIsTraining(false);
              setCurrentJobId(null);
            }}
          />
        )}

        {/* Local training progress */}
        {isTraining && effectiveEnvironment === 'local' && localProgress && (
          <div className="flex items-center space-x-2 p-3 bg-indigo-50 rounded-lg">
            <Loader2 className="w-4 h-4 text-indigo-600 animate-spin" />
            <span className="text-sm text-indigo-700">{localProgress}</span>
          </div>
        )}

        {/* Error display */}
        {error && (
          <div className="flex items-start space-x-2 p-3 bg-red-50 rounded-lg text-sm text-red-700">
            <XCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <div>
              <strong>Error:</strong> {error}
            </div>
          </div>
        )}

        {/* Action button */}
        {!currentJobId && (
          <button
            onClick={handleStartTraining}
            disabled={isTraining || data.length === 0}
            className={`w-full flex items-center justify-center space-x-2 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
              isTraining || data.length === 0
                ? 'bg-gray-400 text-white cursor-not-allowed'
                : effectiveEnvironment === 'cluster'
                  ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
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
                <span>
                  Start Training ({effectiveEnvironment === 'cluster' ? 'Cluster' : 'Local'})
                </span>
              </>
            )}
          </button>
        )}
      </div>
    </div>
  );
};

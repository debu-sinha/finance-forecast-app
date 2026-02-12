import React, { useState, useEffect, useRef } from 'react';
import {
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Play,
  Square,
  RefreshCw,
  ExternalLink,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { TrainingJob, JobStatus } from '../types';
import { logger } from '../utils/logger';
import {
  getJobStatus,
  cancelJob,
  getJobStatusMessage,
  isJobTerminal,
  canCancelJob,
} from '../services/jobApi';

interface JobStatusPanelProps {
  jobId: string;
  onComplete?: (job: TrainingJob) => void;
  onCancel?: () => void;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const getStatusIcon = (status: JobStatus) => {
  switch (status) {
    case 'pending':
      return <Clock className="w-5 h-5 text-gray-500" />;
    case 'submitting':
      return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
    case 'running':
      return <Loader2 className="w-5 h-5 text-indigo-600 animate-spin" />;
    case 'completed':
      return <CheckCircle2 className="w-5 h-5 text-green-500" />;
    case 'failed':
      return <XCircle className="w-5 h-5 text-red-500" />;
    case 'cancelled':
      return <Square className="w-5 h-5 text-orange-500" />;
    case 'cancelling':
      return <Loader2 className="w-5 h-5 text-orange-500 animate-spin" />;
    default:
      return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
  }
};

const getStatusColor = (status: JobStatus) => {
  switch (status) {
    case 'pending':
      return 'bg-gray-100 border-gray-300';
    case 'submitting':
      return 'bg-blue-50 border-blue-300';
    case 'running':
      return 'bg-indigo-50 border-indigo-300';
    case 'completed':
      return 'bg-green-50 border-green-300';
    case 'failed':
      return 'bg-red-50 border-red-300';
    case 'cancelled':
    case 'cancelling':
      return 'bg-orange-50 border-orange-300';
    default:
      return 'bg-gray-50 border-gray-300';
  }
};

export const JobStatusPanel: React.FC<JobStatusPanelProps> = ({
  jobId,
  onComplete,
  onCancel,
  autoRefresh = true,
  refreshInterval = 3000,
}) => {
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch job status
  const fetchStatus = async () => {
    try {
      const jobData = await getJobStatus(jobId);
      setJob(jobData);
      setError(null);

      // Check for completion
      if (isJobTerminal(jobData) && onComplete) {
        onComplete(jobData);
      }

      return jobData;
    } catch (err) {
      logger.error('Failed to fetch job status:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch job status');
      return null;
    }
  };

  // Initial fetch and polling setup
  useEffect(() => {
    fetchStatus();

    if (autoRefresh) {
      intervalRef.current = setInterval(async () => {
        const currentJob = await fetchStatus();
        // Stop polling if job is terminal
        if (currentJob && isJobTerminal(currentJob)) {
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
        }
      }, refreshInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [jobId, autoRefresh, refreshInterval]);

  // Handle cancel
  const handleCancel = async () => {
    if (!job || !canCancelJob(job)) return;

    setIsCancelling(true);
    try {
      const cancelledJob = await cancelJob(jobId);
      setJob(cancelledJob);
      if (onCancel) {
        onCancel();
      }
    } catch (err) {
      logger.error('Failed to cancel job:', err);
      setError(err instanceof Error ? err.message : 'Failed to cancel job');
    } finally {
      setIsCancelling(false);
    }
  };

  if (!job) {
    return (
      <div className="flex items-center justify-center p-4 bg-gray-50 rounded-lg">
        <Loader2 className="w-5 h-5 text-gray-400 animate-spin mr-2" />
        <span className="text-sm text-gray-500">Loading job status...</span>
      </div>
    );
  }

  return (
    <div className={`rounded-lg border-2 ${getStatusColor(job.status)} overflow-hidden`}>
      {/* Header */}
      <div className="px-4 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {getStatusIcon(job.status)}
          <div>
            <div className="text-sm font-medium text-gray-800">
              {getJobStatusMessage(job)}
            </div>
            <div className="text-xs text-gray-500">
              Job ID: {job.job_id.substring(0, 8)}...
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {/* Progress indicator */}
          {job.status === 'running' && (
            <div className="flex items-center space-x-2">
              <div className="w-24 bg-gray-200 rounded-full h-2">
                <div
                  className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${job.progress}%` }}
                />
              </div>
              <span className="text-xs font-medium text-indigo-600">{job.progress}%</span>
            </div>
          )}

          {/* Cancel button */}
          {canCancelJob(job) && (
            <button
              onClick={handleCancel}
              disabled={isCancelling}
              className="flex items-center space-x-1 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-xs font-medium rounded-lg transition-colors disabled:opacity-50"
            >
              {isCancelling ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Square className="w-3 h-3" />
              )}
              <span>Cancel</span>
            </button>
          )}

          {/* Manual refresh button */}
          <button
            onClick={fetchStatus}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-white rounded-lg transition-colors"
            title="Refresh status"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          {/* Toggle details */}
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-white rounded-lg transition-colors"
          >
            {showDetails ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Details panel */}
      {showDetails && (
        <div className="px-4 py-3 border-t border-current border-opacity-20 bg-white bg-opacity-50">
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-gray-500">Status:</span>
              <span className="ml-2 font-medium text-gray-800 capitalize">{job.status}</span>
            </div>
            <div>
              <span className="text-gray-500">Progress:</span>
              <span className="ml-2 font-medium text-gray-800">{job.progress}%</span>
            </div>
            {job.created_at && (
              <div>
                <span className="text-gray-500">Created:</span>
                <span className="ml-2 font-medium text-gray-800">
                  {new Date(job.created_at).toLocaleTimeString()}
                </span>
              </div>
            )}
            {job.submitted_at && (
              <div>
                <span className="text-gray-500">Submitted:</span>
                <span className="ml-2 font-medium text-gray-800">
                  {new Date(job.submitted_at).toLocaleTimeString()}
                </span>
              </div>
            )}
            {job.completed_at && (
              <div>
                <span className="text-gray-500">Completed:</span>
                <span className="ml-2 font-medium text-gray-800">
                  {new Date(job.completed_at).toLocaleTimeString()}
                </span>
              </div>
            )}
            {job.run_id && (
              <div className="col-span-2">
                <span className="text-gray-500">Databricks Run ID:</span>
                <span className="ml-2 font-medium text-gray-800">{job.run_id}</span>
              </div>
            )}
            {job.error && (
              <div className="col-span-2">
                <span className="text-gray-500">Error:</span>
                <span className="ml-2 font-medium text-red-600">{job.error}</span>
              </div>
            )}
          </div>

          {/* Results link for completed jobs */}
          {job.status === 'completed' && job.results?.mlflow_run_id && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <a
                href={`#mlflow-run-${job.results.mlflow_run_id}`}
                className="inline-flex items-center text-sm text-indigo-600 hover:text-indigo-800"
              >
                <ExternalLink className="w-4 h-4 mr-1" />
                View MLflow Run
              </a>
            </div>
          )}
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="px-4 py-2 bg-red-50 border-t border-red-200">
          <div className="flex items-center text-sm text-red-600">
            <AlertTriangle className="w-4 h-4 mr-2" />
            {error}
          </div>
        </div>
      )}
    </div>
  );
};

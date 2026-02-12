/**
 * Job API Service - Manages distributed training jobs on Databricks clusters.
 *
 * This service provides the frontend interface for:
 * - Creating and submitting training jobs
 * - Polling job status with progress updates
 * - Cancelling running jobs
 * - Retrieving job results
 */

import {
    DataRow,
    TrainingMode,
    TrainingJob,
    JobConfig,
    JobResults,
    DelegationStatus,
    TrainingModeInfo,
} from '../types';
import { logFunctionIO, logSyncFunctionIO, logger } from '../utils/logger';

const API_BASE = '/api/v2/jobs';

/**
 * Check if cluster delegation is enabled and get configuration.
 */
const _getDelegationStatus = async (): Promise<DelegationStatus> => {
    const response = await fetch(`${API_BASE}/delegation-status`);

    if (!response.ok) {
        return {
            enabled: false,
            message: 'Failed to check delegation status',
        };
    }

    return response.json();
};
export const getDelegationStatus = logFunctionIO('getDelegationStatus', _getDelegationStatus);

/**
 * Get available training modes with descriptions.
 */
const _getTrainingModes = async (): Promise<TrainingModeInfo[]> => {
    const response = await fetch(`${API_BASE}/training-modes`);

    if (!response.ok) {
        throw new Error('Failed to fetch training modes');
    }

    const data = await response.json();
    return data.modes;
};
export const getTrainingModes = logFunctionIO('getTrainingModes', _getTrainingModes);

/**
 * Create a new training job (does not start it).
 */
const _createJob = async (config: JobConfig): Promise<TrainingJob> => {
    const response = await fetch(API_BASE, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = 'Failed to create job';
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch {
            errDetail = text || errDetail;
        }
        throw new Error(errDetail);
    }

    return response.json();
};
export const createJob = logFunctionIO('createJob', _createJob);

/**
 * Submit a pending job to the Databricks cluster.
 */
const _submitJob = async (jobId: string): Promise<TrainingJob> => {
    const response = await fetch(`${API_BASE}/${jobId}/submit`, {
        method: 'POST',
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = 'Failed to submit job';
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch {
            errDetail = text || errDetail;
        }
        throw new Error(errDetail);
    }

    return response.json();
};
export const submitJob = logFunctionIO('submitJob', _submitJob);

/**
 * Get current job status.
 */
const _getJobStatus = async (jobId: string): Promise<TrainingJob> => {
    const response = await fetch(`${API_BASE}/${jobId}`);

    if (!response.ok) {
        const text = await response.text();
        let errDetail = 'Failed to get job status';
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch {
            errDetail = text || errDetail;
        }
        throw new Error(errDetail);
    }

    return response.json();
};
export const getJobStatus = logFunctionIO('getJobStatus', _getJobStatus);

/**
 * Cancel a running or pending job.
 */
const _cancelJob = async (jobId: string): Promise<TrainingJob> => {
    const response = await fetch(`${API_BASE}/${jobId}/cancel`, {
        method: 'POST',
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = 'Failed to cancel job';
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch {
            errDetail = text || errDetail;
        }
        throw new Error(errDetail);
    }

    return response.json();
};
export const cancelJob = logFunctionIO('cancelJob', _cancelJob);

/**
 * Delete a job from the system.
 */
const _deleteJob = async (jobId: string): Promise<void> => {
    const response = await fetch(`${API_BASE}/${jobId}`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        const text = await response.text();
        let errDetail = 'Failed to delete job';
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch {
            errDetail = text || errDetail;
        }
        throw new Error(errDetail);
    }
};
export const deleteJob = logFunctionIO('deleteJob', _deleteJob);

/**
 * List all jobs with optional status filter.
 */
const _listJobs = async (
    status?: string,
    limit: number = 50
): Promise<{ jobs: TrainingJob[]; total: number }> => {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('limit', String(limit));

    const response = await fetch(`${API_BASE}?${params.toString()}`);

    if (!response.ok) {
        throw new Error('Failed to list jobs');
    }

    return response.json();
};
export const listJobs = logFunctionIO('listJobs', _listJobs);

/**
 * Get results from a completed job.
 */
const _getJobResults = async (jobId: string): Promise<JobResults> => {
    const response = await fetch(`${API_BASE}/${jobId}/results`);

    if (!response.ok) {
        const text = await response.text();
        let errDetail = 'Failed to get job results';
        try {
            const json = JSON.parse(text);
            errDetail = json.detail || errDetail;
        } catch {
            errDetail = text || errDetail;
        }
        throw new Error(errDetail);
    }

    return response.json();
};
export const getJobResults = logFunctionIO('getJobResults', _getJobResults);

// ==========================================
// JOB POLLING UTILITIES
// ==========================================

export interface JobPollingOptions {
    /** Polling interval in milliseconds (default: 3000) */
    intervalMs?: number;
    /** Maximum polling duration in milliseconds (default: 3600000 = 1 hour) */
    maxDurationMs?: number;
    /** Callback for progress updates */
    onProgress?: (job: TrainingJob) => void;
    /** Callback for errors */
    onError?: (error: Error) => void;
    /** AbortSignal for cancellation */
    signal?: AbortSignal;
}

/**
 * Poll job status until completion or failure.
 * Returns the final job state.
 */
const _pollJobUntilComplete = async (
    jobId: string,
    options: JobPollingOptions = {}
): Promise<TrainingJob> => {
    const {
        intervalMs = 3000,
        maxDurationMs = 3600000,
        onProgress,
        onError,
        signal,
    } = options;

    const startTime = Date.now();

    return new Promise((resolve, reject) => {
        const poll = async () => {
            // Check for abort
            if (signal?.aborted) {
                reject(new Error('Polling cancelled'));
                return;
            }

            // Check for timeout
            if (Date.now() - startTime > maxDurationMs) {
                reject(new Error('Job polling timed out'));
                return;
            }

            try {
                const job = await _getJobStatus(jobId);

                // Notify progress
                if (onProgress) {
                    onProgress(job);
                }

                // Check terminal states
                if (['completed', 'failed', 'cancelled'].includes(job.status)) {
                    resolve(job);
                    return;
                }

                // Continue polling
                setTimeout(poll, intervalMs);
            } catch (error) {
                if (onError) {
                    onError(error as Error);
                }
                reject(error);
            }
        };

        // Start polling
        poll();
    });
};
export const pollJobUntilComplete = logFunctionIO('pollJobUntilComplete', _pollJobUntilComplete);

// ==========================================
// CONVENIENCE FUNCTIONS
// ==========================================

/**
 * Create and immediately submit a job, then poll for completion.
 * This is the most common workflow for users.
 */
const _runDistributedTraining = async (
    data: DataRow[],
    config: {
        time_col: string;
        target_col: string;
        id_col?: string;
        covariates: string[];
        horizon: number;
        frequency: string;
        training_mode: TrainingMode;
        models?: string[];
        seasonality_mode?: string;
        time_limit?: number;
        presets?: string;
        season_length?: number;
    },
    options: JobPollingOptions = {}
): Promise<TrainingJob> => {
    // Create the job config
    const jobConfig: JobConfig = {
        data,
        time_col: config.time_col,
        target_col: config.target_col,
        id_col: config.id_col,
        covariates: config.covariates,
        horizon: config.horizon,
        frequency: config.frequency,
        training_mode: config.training_mode,
        models: config.models || ['prophet'],
        seasonality_mode: config.seasonality_mode || 'multiplicative',
        time_limit: config.time_limit || 600,
        presets: config.presets || 'medium_quality',
        season_length: config.season_length,
    };

    // Create the job
    const job = await _createJob(jobConfig);

    // Submit immediately
    const submittedJob = await _submitJob(job.job_id);

    // Poll for completion
    return _pollJobUntilComplete(submittedJob.job_id, options);
};
export const runDistributedTraining = logFunctionIO('runDistributedTraining', _runDistributedTraining);

/**
 * Get a human-readable status message for a job.
 */
const _getJobStatusMessage = (job: TrainingJob): string => {
    switch (job.status) {
        case 'pending':
            return 'Job created, waiting to start';
        case 'submitting':
            return 'Submitting to Databricks cluster...';
        case 'running':
            return job.current_step || 'Training in progress...';
        case 'completed':
            return 'Training completed successfully';
        case 'failed':
            return job.error || 'Training failed';
        case 'cancelled':
            return 'Job was cancelled';
        case 'cancelling':
            return 'Cancelling job...';
        default:
            return 'Unknown status';
    }
};
export const getJobStatusMessage = logSyncFunctionIO('getJobStatusMessage', _getJobStatusMessage);

/**
 * Check if a job is in a terminal state.
 */
const _isJobTerminal = (job: TrainingJob): boolean => {
    return ['completed', 'failed', 'cancelled'].includes(job.status);
};
export const isJobTerminal = logSyncFunctionIO('isJobTerminal', _isJobTerminal);

/**
 * Check if a job can be cancelled.
 */
const _canCancelJob = (job: TrainingJob): boolean => {
    return ['pending', 'submitting', 'running'].includes(job.status);
};
export const canCancelJob = logSyncFunctionIO('canCancelJob', _canCancelJob);

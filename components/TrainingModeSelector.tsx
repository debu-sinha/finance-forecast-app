import React, { useState, useEffect } from 'react';
import {
  Zap,
  Cpu,
  Brain,
  BarChart3,
  Clock,
  ChevronDown,
  Check,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import { TrainingMode, TrainingModeInfo } from '../types';
import { getTrainingModes, getDelegationStatus } from '../services/jobApi';

interface TrainingModeSelectorProps {
  selectedMode: TrainingMode;
  onModeChange: (mode: TrainingMode) => void;
  disabled?: boolean;
}

const getModeIcon = (mode: TrainingMode) => {
  switch (mode) {
    case 'autogluon':
      return <Cpu className="w-4 h-4" />;
    case 'statsforecast':
      return <Zap className="w-4 h-4" />;
    case 'neuralforecast':
      return <Brain className="w-4 h-4" />;
    case 'mmf':
      return <BarChart3 className="w-4 h-4" />;
    case 'legacy':
      return <Clock className="w-4 h-4" />;
    default:
      return <Cpu className="w-4 h-4" />;
  }
};

const getSpeedBadgeColor = (speed: string) => {
  switch (speed) {
    case 'fast':
      return 'bg-green-100 text-green-700';
    case 'medium':
      return 'bg-yellow-100 text-yellow-700';
    case 'slow':
      return 'bg-orange-100 text-orange-700';
    case 'variable':
      return 'bg-purple-100 text-purple-700';
    default:
      return 'bg-gray-100 text-gray-700';
  }
};

export const TrainingModeSelector: React.FC<TrainingModeSelectorProps> = ({
  selectedMode,
  onModeChange,
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [modes, setModes] = useState<TrainingModeInfo[]>([]);
  const [delegationEnabled, setDelegationEnabled] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch available modes and delegation status on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [modesData, statusData] = await Promise.all([
          getTrainingModes(),
          getDelegationStatus(),
        ]);
        setModes(modesData);
        setDelegationEnabled(statusData.enabled);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch training modes:', err);
        setError('Failed to load training modes');
        // Set default modes if fetch fails
        setModes([
          {
            value: 'autogluon',
            name: 'AutoGluon-TimeSeries',
            description: 'Best accuracy with automatic ensembling',
            speed: 'medium',
            recommended: true,
          },
          {
            value: 'legacy',
            name: 'Legacy (Prophet/ARIMA)',
            description: 'Original implementation',
            speed: 'medium',
            recommended: false,
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const selectedModeInfo = modes.find(m => m.value === selectedMode);

  if (loading) {
    return (
      <div className="flex items-center space-x-2 px-3 py-2 bg-gray-100 rounded-lg">
        <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
        <span className="text-sm text-gray-500">Loading training modes...</span>
      </div>
    );
  }

  if (!delegationEnabled) {
    return (
      <div className="flex items-center space-x-2 px-3 py-2 bg-yellow-50 border border-yellow-200 rounded-lg">
        <AlertCircle className="w-4 h-4 text-yellow-600" />
        <span className="text-sm text-yellow-700">
          Cluster training not available. Using local training.
        </span>
      </div>
    );
  }

  return (
    <div className="relative">
      <label className="block text-xs font-medium text-gray-600 mb-1">
        Training Mode (Cluster)
      </label>
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`w-full flex items-center justify-between px-3 py-2 bg-white border rounded-lg text-left transition-colors ${
          disabled
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed'
            : 'border-gray-300 hover:border-indigo-400 cursor-pointer'
        }`}
      >
        <div className="flex items-center space-x-2">
          <span className="text-indigo-600">{getModeIcon(selectedMode)}</span>
          <div>
            <span className="text-sm font-medium text-gray-800">
              {selectedModeInfo?.name || selectedMode}
            </span>
            {selectedModeInfo?.recommended && (
              <span className="ml-2 px-1.5 py-0.5 bg-indigo-100 text-indigo-700 text-xs rounded-full">
                Recommended
              </span>
            )}
          </div>
        </div>
        <ChevronDown
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden">
          {modes.map(mode => (
            <button
              key={mode.value}
              onClick={() => {
                onModeChange(mode.value);
                setIsOpen(false);
              }}
              className={`w-full flex items-start px-3 py-3 text-left hover:bg-gray-50 transition-colors ${
                selectedMode === mode.value ? 'bg-indigo-50' : ''
              }`}
            >
              <span className={`mt-0.5 mr-3 ${selectedMode === mode.value ? 'text-indigo-600' : 'text-gray-400'}`}>
                {getModeIcon(mode.value)}
              </span>
              <div className="flex-1">
                <div className="flex items-center">
                  <span className={`text-sm font-medium ${selectedMode === mode.value ? 'text-indigo-700' : 'text-gray-800'}`}>
                    {mode.name}
                  </span>
                  {mode.recommended && (
                    <span className="ml-2 px-1.5 py-0.5 bg-indigo-100 text-indigo-700 text-xs rounded-full">
                      Recommended
                    </span>
                  )}
                  <span className={`ml-2 px-1.5 py-0.5 text-xs rounded-full ${getSpeedBadgeColor(mode.speed)}`}>
                    {mode.speed}
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-0.5">{mode.description}</p>
              </div>
              {selectedMode === mode.value && (
                <Check className="w-4 h-4 text-indigo-600 mt-0.5" />
              )}
            </button>
          ))}
        </div>
      )}

      {error && (
        <p className="mt-1 text-xs text-red-500">{error}</p>
      )}
    </div>
  );
};

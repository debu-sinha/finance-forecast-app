import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { DataRow } from '../types';

/**
 * Parse a date string without timezone issues.
 * Adding 'T12:00:00' ensures we're at noon UTC, avoiding midnight boundary issues
 * that can cause dates to shift when displayed in local timezone.
 */
const parseDate = (dateStr: string): Date => {
  if (!dateStr) return new Date(NaN);
  // If it's already a full ISO string with time, use as-is
  if (dateStr.includes('T')) {
    return new Date(dateStr);
  }
  // For date-only strings (YYYY-MM-DD), add noon time to avoid timezone boundary issues
  return new Date(dateStr + 'T12:00:00');
};

interface EvaluationChartProps {
  validation: DataRow[];
  timeCol: string;
  targetCol: string;
}

export const EvaluationChart: React.FC<EvaluationChartProps> = ({
  validation,
  timeCol,
  targetCol
}) => {
  if (!validation || validation.length === 0) {
    return <div className="flex items-center justify-center h-full text-gray-400">No validation data available</div>;
  }

  // Debug logging
  console.log('Evaluation Chart Data:', {
    validationLength: validation.length,
    validationSample: validation[0],
    timeCol,
    targetCol
  });

  // Prepare chart data - validation data has 'y' for actuals, 'yhat' for predictions
  const chartData = validation.map(row => {
    // Backend sends: timeCol (renamed from 'ds'), 'y' (actual), 'yhat' (predicted)
    // In batch results, the data might use 'ds' directly or the user's timeCol
    const actual = row['y'] !== undefined && row['y'] !== null
      ? Number(row['y'])
      : (row[targetCol] !== undefined && row[targetCol] !== null
        ? Number(row[targetCol])
        : null);
    const predicted = row['yhat'] !== undefined && row['yhat'] !== null
      ? Number(row['yhat'])
      : null;

    // Try multiple possible date column names
    const dateValue = row[timeCol] || row['ds'] || row['date'] || row['Date'] || '';

    return {
      date: String(dateValue),
      actual: actual,
      predicted: predicted,
      lower: Number(row['yhat_lower'] || 0),
      upper: Number(row['yhat_upper'] || 0)
    };
  }).filter(row => row.date && row.actual !== null && row.predicted !== null && row.actual > 0 && row.predicted > 0); // Filter out invalid data

  if (chartData.length === 0) {
    return <div className="flex items-center justify-center h-full text-gray-400">No valid validation data to display</div>;
  }

  // Calculate accuracy metrics for display
  const calculateAccuracy = () => {
    let totalError = 0;
    chartData.forEach(point => {
      const percentError = Math.abs((point.actual - point.predicted) / point.actual) * 100;
      totalError += percentError;
    });
    return (100 - (totalError / chartData.length)).toFixed(1);
  };

  const accuracy = calculateAccuracy();

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex justify-between items-center mb-3">
        <div>
          <h4 className="text-sm font-semibold text-gray-800">Validation Performance</h4>
          <p className="text-xs text-gray-500">Actual vs Predicted on holdout set</p>
        </div>
        <div className="bg-green-50 border border-green-200 px-3 py-1 rounded">
          <span className="text-xs text-green-700 font-semibold">Accuracy: {accuracy}%</span>
        </div>
      </div>

      <div className="flex-1 min-h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 11, fill: '#666' }}
              tickLine={false}
              axisLine={{ stroke: '#e0e0e0' }}
              tickFormatter={(str) => {
                try {
                  const d = parseDate(str);
                  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                } catch (e) { return str; }
              }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: '#666' }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => value >= 1000 ? `${(value / 1000).toFixed(0)}k` : String(value)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#fff',
                borderRadius: '8px',
                border: '1px solid #e0e0e0',
                boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'
              }}
              itemStyle={{ fontSize: '12px' }}
              labelStyle={{ fontSize: '12px', color: '#888', marginBottom: '4px', fontWeight: 600 }}
              formatter={(value: number, name: string) => {
                const formatted = value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 6 });
                if (name === 'actual') return [formatted, 'Actual Value'];
                if (name === 'predicted') return [formatted, 'Model Prediction'];
                return [formatted, name];
              }}
              labelFormatter={(label) => {
                try {
                  const d = parseDate(label);
                  return d.toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
                } catch (e) { return label; }
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }}
              iconType="line"
            />

            {/* Actual Values */}
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#374151"
              strokeWidth={3}
              dot={{ r: 4, fill: '#374151', strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 6 }}
              name="Actual Value"
            />

            {/* Predicted Values */}
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#f97316"
              strokeWidth={3}
              dot={{ r: 4, fill: '#f97316', strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 6 }}
              name="Model Prediction"
              strokeDasharray="5 5"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Error Analysis */}
      <div className="mt-3 grid grid-cols-3 gap-2">
        {chartData.slice(0, 3).map((point, idx) => {
          const error = ((point.predicted - point.actual) / point.actual * 100).toFixed(1);
          const isGood = Math.abs(parseFloat(error)) < 10;
          return (
            <div key={idx} className="text-center p-2 bg-gray-50 rounded border border-gray-200">
              <div className="text-[10px] text-gray-500">{point.date}</div>
              <div className={`text-xs font-semibold ${isGood ? 'text-green-600' : 'text-orange-600'}`}>
                {error}% error
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

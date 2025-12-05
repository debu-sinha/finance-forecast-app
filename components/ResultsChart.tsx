
import React, { useMemo, useState, useCallback } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  ReferenceArea
} from 'recharts';
import { DataRow } from '../types';

// Columns to exclude from event/holiday detection (system columns)
const EXCLUDED_COLUMNS = [
  'ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper',
  'actual', 'forecast_yhat', 'validation_yhat',
  'type', 'trend', 'trend_lower', 'trend_upper',
  'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
  'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper',
  'yearly', 'yearly_lower', 'yearly_upper',
  'weekly', 'weekly_lower', 'weekly_upper',
  'daily', 'daily_lower', 'daily_upper',
  'holidays', 'holidays_lower', 'holidays_upper'
];

interface ComparisonForecast {
  name: string;
  data: DataRow[];
  color: string;
}

interface ResultsChartProps {
  history: DataRow[];
  validation?: DataRow[];
  forecast: DataRow[];
  timeCol: string;
  targetCol: string;
  showForecast?: boolean;
  comparisonForecasts?: ComparisonForecast[]; // New prop for multi-model comparison
  covariates?: string[];
}

// Custom tooltip component to show holiday/event values
const CustomTooltip = ({ active, payload, label, holidayColumns }: any) => {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload || {};

  // Find active holidays for this data point
  const activeHolidays = holidayColumns.filter((col: string) => {
    const val = data[col];
    return val === 1 || val === '1' || val === true || val === 'true';
  });

  // Format date
  let formattedDate = label;
  try {
    const d = new Date(label);
    if (!isNaN(d.getTime())) {
      formattedDate = d.toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
    }
  } catch (e) { /* keep original */ }

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-lg p-3 max-w-xs">
      <p className="text-xs text-gray-500 mb-2 font-medium">{formattedDate}</p>

      {/* Main values */}
      <div className="space-y-1">
        {payload.map((entry: any, index: number) => {
          if (entry.value === null || entry.value === undefined) return null;
          let displayName = entry.name;
          if (entry.dataKey === 'actual') displayName = 'Historical Actuals';
          else if (entry.dataKey === 'validation_yhat') displayName = 'Model Fit (Eval)';
          else if (entry.dataKey === 'forecast_yhat') displayName = 'Final Forecast';
          else if (entry.dataKey?.startsWith('comp_')) displayName = entry.dataKey.replace('comp_', '');

          return (
            <div key={index} className="flex justify-between items-center text-xs">
              <span style={{ color: entry.color }} className="font-medium">{displayName}</span>
              <span className="font-mono ml-4">{Number(entry.value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 6})}</span>
            </div>
          );
        })}
      </div>

      {/* Holiday/Event indicators */}
      {activeHolidays.length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-100">
          <p className="text-[10px] text-gray-500 uppercase font-semibold mb-1">Events/Holidays</p>
          <div className="flex flex-wrap gap-1">
            {activeHolidays.map((holiday: string) => (
              <span key={holiday} className="text-[10px] bg-amber-100 text-amber-800 px-1.5 py-0.5 rounded font-medium">
                {holiday}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export const ResultsChart: React.FC<ResultsChartProps> = ({
  history,
  validation,
  forecast,
  timeCol,
  targetCol,
  showForecast = true,
  comparisonForecasts = [],
  covariates = []
}) => {
  // Zoom state
  const [zoomLeft, setZoomLeft] = useState<string | null>(null);
  const [zoomRight, setZoomRight] = useState<string | null>(null);
  const [refAreaLeft, setRefAreaLeft] = useState<string>('');
  const [refAreaRight, setRefAreaRight] = useState<string>('');
  const [isSelecting, setIsSelecting] = useState(false);

  // Debug logging
  console.log('Chart Data:', {
    history: history.length,
    validation: validation?.length || 0,
    forecast: forecast.length,
    timeCol,
    targetCol,
    forecastSample: forecast[0],
    validationSample: validation?.[0],
    historySample: history[0],
    lastHistory: history[history.length - 1]
  });

  // Merge data sources into a single timeline
  const chartData = useMemo(() => {
    const dataMap = new Map<string, any>();

    // 1. Add History (Actuals) - Include all columns for holiday detection
    history.forEach(row => {
      const date = String(row[timeCol]);
      // Copy all row data to preserve holiday/covariate columns
      const rowData: any = { ...row };
      rowData[timeCol] = date;
      rowData.actual = Number(row[targetCol]);
      rowData.type = 'history';
      dataMap.set(date, rowData);
    });

    // 2. Add Validation (Model Fit) - Overlay on history (Only for active model)
    // Validation data has 'y' (actual) and 'yhat' (predicted)
    if (validation && validation.length > 0) {
      validation.forEach(row => {
        const date = String(row[timeCol]);
        const existing = dataMap.get(date);
        if (existing) {
          // Use predicted value from validation
          dataMap.set(date, {
            ...existing,
            validation_yhat: Number(row['yhat'] || 0)
          });
        } else {
          // If validation date doesn't exist in history, add it
          dataMap.set(date, {
            [timeCol]: date,
            actual: Number(row['y'] || row[targetCol] || 0),
            validation_yhat: Number(row['yhat'] || 0),
            type: 'validation'
          });
        }
      });
    }

    // Helper to anchor forecast lines to the last history point
    const anchorForecast = (dateKey: string, val: number, keyName: string) => {
      const existing = dataMap.get(dateKey);
      if (existing) {
        dataMap.set(dateKey, { ...existing, [keyName]: val });
      } else {
        // Create entry if it doesn't exist
        dataMap.set(dateKey, {
          [timeCol]: dateKey,
          actual: val, // Use actual value from history
          [keyName]: val
        });
      }
    };

    const lastHistoryRow = history.length > 0 ? history[history.length - 1] : null;
    const lastDate = lastHistoryRow ? String(lastHistoryRow[timeCol]) : null;
    const lastVal = lastHistoryRow ? Number(lastHistoryRow[targetCol]) : 0;
    const firstForecastDate = forecast.length > 0 ? String(forecast[0][timeCol]) : null;
    const firstForecastVal = forecast.length > 0 ? Number(forecast[0]['yhat']) : null;

    // 3. Add Main Forecast (Future)
    if (showForecast && forecast.length > 0) {
      // Anchor main forecast to last history point for smooth transition
      if (lastDate && firstForecastDate && lastDate !== firstForecastDate) {
        anchorForecast(lastDate, lastVal, 'forecast_yhat');
        // Also anchor to first forecast point if there's a gap
        if (firstForecastVal !== null) {
          const existing = dataMap.get(firstForecastDate);
          if (existing) {
            dataMap.set(firstForecastDate, { ...existing, forecast_yhat: firstForecastVal });
          } else {
            dataMap.set(firstForecastDate, {
              [timeCol]: firstForecastDate,
              forecast_yhat: firstForecastVal,
              type: 'forecast'
            });
          }
        }
      }

      forecast.forEach(row => {
        const date = String(row[timeCol]);
        const existing = dataMap.get(date) || { [timeCol]: date };
        const yhat = Number(row['yhat'] || 0);
        // Merge all row data to preserve holiday/covariate columns from forecast
        dataMap.set(date, {
          ...existing,
          ...row, // Include all columns from forecast row
          [timeCol]: date,
          forecast_yhat: yhat,
          yhat_upper: Number(row['yhat_upper'] || yhat * 1.2),
          yhat_lower: Number(row['yhat_lower'] || yhat * 0.8),
          type: 'forecast'
        });
      });
    }

    // 4. Add Comparison Forecasts
    comparisonForecasts.forEach((comp) => {
      const keyName = `comp_${comp.name}`; // Unique key for this model line

      // Anchor comparison line
      if (lastDate) anchorForecast(lastDate, lastVal, keyName);

      comp.data.forEach(row => {
        const date = String(row[timeCol]);
        const existing = dataMap.get(date) || { [timeCol]: date };
        dataMap.set(date, {
          ...existing,
          [keyName]: Number(row['yhat'])
        });
      });
    });

    // 5. Add Covariate Values (for overlays)
    // We need to ensure covariate values are present in the dataMap
    // They should already be there if they were in history/validation/forecast data
    // But let's make sure we extract them properly if they are in the row

    return Array.from(dataMap.values()).sort((a, b) =>
      new Date(a[timeCol]).getTime() - new Date(b[timeCol]).getTime()
    );
  }, [history, validation, forecast, timeCol, targetCol, showForecast, comparisonForecasts, covariates]);

  const lastHistoryDate = history.length > 0 ? String(history[history.length - 1][timeCol]) : null;

  // Dynamically detect event/promo columns (binary 0/1 columns)
  const holidayColumns = useMemo(() => {
    if (chartData.length === 0) return [];

    // Get all unique keys from the data
    const allKeys = new Set<string>();
    chartData.forEach(row => Object.keys(row).forEach(key => allKeys.add(key)));

    const detected: string[] = [];

    // Check each column to see if it's a binary event column
    allKeys.forEach(key => {
      // Skip excluded system columns and the time/target columns
      if (EXCLUDED_COLUMNS.includes(key) || key === timeCol || key === targetCol) return;
      // Skip columns that start with 'comp_' (comparison forecasts)
      if (key.startsWith('comp_')) return;

      // Check if this column has binary (0/1) values - indicating it's an event flag
      let hasZero = false;
      let hasOne = false;
      let hasOtherValues = false;

      for (const row of chartData) {
        const val = row[key];
        if (val === undefined || val === null || val === '') continue;

        const numVal = Number(val);
        if (numVal === 0 || val === '0' || val === false) {
          hasZero = true;
        } else if (numVal === 1 || val === '1' || val === true) {
          hasOne = true;
        } else if (!isNaN(numVal) && numVal !== 0 && numVal !== 1) {
          // Has non-binary numeric values - not an event column
          hasOtherValues = true;
          break;
        }
      }

      // It's an event column if it only has 0/1 values and has at least one "1" (active event)
      if (hasOne && !hasOtherValues) {
        detected.push(key);
      }
    });

    // Also include covariates that might be events
    covariates.forEach(cov => {
      if (!detected.includes(cov) && !EXCLUDED_COLUMNS.includes(cov)) {
        // Check if covariate has any "1" values
        const hasActiveEvent = chartData.some(row => {
          const val = row[cov];
          return val === 1 || val === '1' || val === true;
        });
        if (hasActiveEvent) {
          detected.push(cov);
        }
      }
    });

    console.log('Dynamically detected event/promo columns:', detected);
    return detected;
  }, [chartData, covariates, timeCol, targetCol]);

  // Zoom handlers
  const handleMouseDown = useCallback((e: any) => {
    if (e && e.activeLabel) {
      setRefAreaLeft(e.activeLabel);
      setIsSelecting(true);
    }
  }, []);

  const handleMouseMove = useCallback((e: any) => {
    if (isSelecting && e && e.activeLabel) {
      setRefAreaRight(e.activeLabel);
    }
  }, [isSelecting]);

  const handleMouseUp = useCallback(() => {
    if (refAreaLeft && refAreaRight) {
      // Ensure left < right
      const leftIdx = chartData.findIndex(d => d[timeCol] === refAreaLeft);
      const rightIdx = chartData.findIndex(d => d[timeCol] === refAreaRight);

      if (leftIdx !== -1 && rightIdx !== -1) {
        if (leftIdx <= rightIdx) {
          setZoomLeft(refAreaLeft);
          setZoomRight(refAreaRight);
        } else {
          setZoomLeft(refAreaRight);
          setZoomRight(refAreaLeft);
        }
      }
    }
    setRefAreaLeft('');
    setRefAreaRight('');
    setIsSelecting(false);
  }, [refAreaLeft, refAreaRight, chartData, timeCol]);

  const handleZoomOut = useCallback(() => {
    setZoomLeft(null);
    setZoomRight(null);
  }, []);

  // Filter data based on zoom
  const displayData = useMemo(() => {
    if (!zoomLeft || !zoomRight) return chartData;

    const leftIdx = chartData.findIndex(d => d[timeCol] === zoomLeft);
    const rightIdx = chartData.findIndex(d => d[timeCol] === zoomRight);

    if (leftIdx === -1 || rightIdx === -1) return chartData;

    return chartData.slice(Math.max(0, leftIdx - 1), Math.min(chartData.length, rightIdx + 2));
  }, [chartData, zoomLeft, zoomRight, timeCol]);

  if (!targetCol || !timeCol) return null;

  // Heuristic: If we have many points (> 30), don't show dots to keep the line clean
  const showDots = displayData.length < 30;
  const isZoomed = zoomLeft !== null && zoomRight !== null;

  return (
    <div className="w-full h-[400px] relative">
      {/* Zoom controls */}
      <div className="absolute top-0 right-0 z-10 flex items-center space-x-2">
        {isZoomed && (
          <button
            onClick={handleZoomOut}
            className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-2 py-1 rounded border border-gray-300 flex items-center"
          >
            <span className="mr-1">↩</span> Reset Zoom
          </button>
        )}
        {holidayColumns.length > 0 && (
          <span className="text-[10px] text-amber-600 bg-amber-50 px-2 py-1 rounded border border-amber-200">
            {holidayColumns.length} event{holidayColumns.length !== 1 ? 's' : ''} detected
          </span>
        )}
      </div>
      <p className="text-[10px] text-gray-400 absolute top-0 left-0">Drag to zoom • Hover for details</p>

      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={displayData}
          margin={{ top: 30, right: 20, bottom: 40, left: 20 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
          <XAxis
            dataKey={timeCol}
            tick={{ fontSize: 11, fill: '#666' }}
            tickLine={false}
            axisLine={{ stroke: '#e0e0e0' }}
            minTickGap={40}
            tickFormatter={(str) => {
              try {
                const d = new Date(str);
                if (isNaN(d.getTime())) return str;
                return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
              } catch (e) { return str; }
            }}
            allowDataOverflow={true}
          />
          <YAxis
            tick={{ fontSize: 11, fill: '#666' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(value)}
            domain={['auto', 'auto']}
          />
          <Tooltip content={<CustomTooltip holidayColumns={holidayColumns} />} />
          <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />

          {showForecast && lastHistoryDate && (
            <ReferenceLine x={lastHistoryDate} stroke="#9ca3af" strokeDasharray="3 3" label={{ value: "Forecast Start", position: 'insideTopRight', fill: '#9ca3af', fontSize: 10 }} />
          )}

          {/* Covariate Overlays - Show in Tooltip Only */}
          {/* Covariates are now part of chartData and will appear in tooltip automatically */}

          {/* Confidence Intervals (Only for Active Model) */}
          {showForecast && (
            <>
              <Area
                type="monotone"
                dataKey="yhat_upper"
                stroke="none"
                fill="#7c3aed"
                fillOpacity={0.1}
                legendType="none"
                tooltipType="none"
                connectNulls
              />
              <Area
                type="monotone"
                dataKey="yhat_lower"
                stroke="none"
                fill="#7c3aed"
                fillOpacity={0.1}
                legendType="none"
                tooltipType="none"
                connectNulls
              />
            </>
          )}

          {/* 1. Historical Actuals */}
          <Line
            type="monotone"
            dataKey="actual"
            stroke="#374151"
            strokeWidth={2}
            dot={showDots ? { r: 3, fill: '#374151' } : false}
            activeDot={{ r: 5 }}
            name="Historical Actuals"
            connectNulls
          />

          {/* 2. Model Fit (Validation) - Orange dashed line */}
          {validation && validation.length > 0 && (
            <Line
              type="monotone"
              dataKey="validation_yhat"
              stroke="#f97316"
              strokeWidth={3}
              strokeDasharray="5 5"
              dot={{ r: 5, fill: '#f97316', strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 7 }}
              name="Model Fit (Eval)"
              connectNulls={false}
            />
          )}

          {/* 3. Future Forecast (Active) */}
          {showForecast && (
            <Line
              type="monotone"
              dataKey="forecast_yhat"
              stroke="#7c3aed"
              strokeWidth={3}
              strokeDasharray="6 4"
              dot={{ r: 5, fill: '#7c3aed', strokeWidth: 2, stroke: '#fff' }}
              activeDot={{ r: 7 }}
              name="Final Forecast"
              connectNulls={false}
            />
          )}

          {/* 4. Comparison Lines (Dynamic) */}
          {comparisonForecasts.map((comp) => (
            <Line
              key={comp.name}
              type="monotone"
              dataKey={`comp_${comp.name}`}
              stroke={comp.color}
              strokeWidth={2}
              strokeDasharray="2 2"
              dot={false}
              name={comp.name}
              connectNulls
            />
          ))}

          {/* Zoom selection area */}
          {refAreaLeft && refAreaRight && (
            <ReferenceArea
              x1={refAreaLeft}
              x2={refAreaRight}
              strokeOpacity={0.3}
              fill="#3b82f6"
              fillOpacity={0.2}
            />
          )}

          {/* Brush for timeline navigation */}
          {!isZoomed && chartData.length > 20 && (
            <Brush
              dataKey={timeCol}
              height={25}
              stroke="#9ca3af"
              fill="#f9fafb"
              tickFormatter={(str) => {
                try {
                  const d = new Date(str);
                  if (isNaN(d.getTime())) return '';
                  return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
                } catch (e) { return ''; }
              }}
            />
          )}

        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

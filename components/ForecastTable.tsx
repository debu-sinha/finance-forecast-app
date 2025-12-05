import React, { useState } from 'react';
import { DataRow } from '../types';
import { FileDown, ArrowUpCircle, ArrowDownCircle, TrendingUp } from 'lucide-react';

interface ForecastTableProps {
  forecast: DataRow[];
  timeCol: string;
  targetCol: string;
}

export const ForecastTable: React.FC<ForecastTableProps> = ({ forecast, timeCol, targetCol }) => {
  const [showAll, setShowAll] = useState(false);
  
  if (!forecast || forecast.length === 0) return <div className="text-gray-400 text-sm">No forecast data available</div>;

  const displayData = showAll ? forecast : forecast.slice(0, 12);
  
  // Calculate period-over-period growth
  const calcGrowth = (idx: number) => {
    if (idx === 0) return null;
    const current = Number(forecast[idx]['yhat'] || forecast[idx][targetCol]);
    const previous = Number(forecast[idx - 1]['yhat'] || forecast[idx - 1][targetCol]);
    if (previous === 0) return null;
    return ((current - previous) / previous * 100).toFixed(1);
  };

  const downloadCSV = () => {
    const headers = [timeCol, 'Forecast', 'Lower 95% CI', 'Upper 95% CI', 'Change %'];
    const csvContent = [
      headers.join(','),
      ...forecast.map((row, idx) => {
        const growth = calcGrowth(idx);
        return [
          String(row[timeCol]),
          Number(row['yhat'] || row[targetCol]).toFixed(6),
          Number(row['yhat_lower'] || 0).toFixed(6),
          Number(row['yhat_upper'] || 0).toFixed(6),
          growth || 'N/A'
        ].join(',');
      })
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `forecast_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <button
          onClick={downloadCSV}
          className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded flex items-center space-x-1.5 transition-colors shadow-sm"
        >
          <FileDown className="w-3.5 h-3.5" />
          <span>Export CSV</span>
        </button>
      </div>
      
      <div className="overflow-auto max-h-96 border border-gray-200 rounded-lg">
        <table className="w-full text-sm">
          <thead className="bg-gradient-to-r from-blue-50 to-indigo-50 sticky top-0 z-10">
            <tr>
              <th className="px-4 py-3 text-left font-semibold text-gray-700 border-b-2 border-blue-200">Period</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-700 border-b-2 border-blue-200">Forecast</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-700 text-xs border-b-2 border-blue-200">Change</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-700 text-xs border-b-2 border-blue-200">Lower 95%</th>
              <th className="px-4 py-3 text-right font-semibold text-gray-700 text-xs border-b-2 border-blue-200">Upper 95%</th>
              <th className="px-4 py-3 text-center font-semibold text-gray-700 text-xs border-b-2 border-blue-200">Uncertainty</th>
            </tr>
          </thead>
          <tbody>
            {displayData.map((row, idx) => {
              const forecastVal = Number(row['yhat'] || row[targetCol]);
              const lowerVal = Number(row['yhat_lower'] || 0);
              const upperVal = Number(row['yhat_upper'] || 0);
              const growth = calcGrowth(idx);
              const range = upperVal - lowerVal;
              const uncertainty = range > 0 ? ((range / forecastVal) * 100).toFixed(1) : '0';
              
              return (
                <tr key={idx} className="border-t border-gray-100 hover:bg-blue-50/40 transition-colors">
                  <td className="px-4 py-2.5 text-gray-900 font-medium whitespace-nowrap">
                    {String(row[timeCol])}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono font-bold text-blue-700 text-base">
                    {forecastVal.toLocaleString(undefined, {minimumFractionDigits: 4, maximumFractionDigits: 6})}
                  </td>
                  <td className="px-4 py-2.5 text-right">
                    {growth !== null ? (
                      <span className={`flex items-center justify-end space-x-1 text-xs font-semibold ${parseFloat(growth) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {parseFloat(growth) >= 0 ? <ArrowUpCircle className="w-3 h-3" /> : <ArrowDownCircle className="w-3 h-3" />}
                        <span>{Math.abs(parseFloat(growth))}%</span>
                      </span>
                    ) : <span className="text-xs text-gray-400">-</span>}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-500">
                    {lowerVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4})}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-500">
                    {upperVal.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4})}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    <span className="inline-block px-2 py-0.5 bg-gray-100 text-gray-700 rounded text-xs font-medium">
                      ±{uncertainty}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      
      {forecast.length > 12 && (
        <div className="flex justify-center">
          <button
            onClick={() => setShowAll(!showAll)}
            className="text-xs text-blue-600 hover:text-blue-700 py-2 px-4 font-medium hover:bg-blue-50 rounded transition-colors"
          >
            {showAll ? `Show less` : `Show all ${forecast.length} periods`}
          </button>
        </div>
      )}
    </div>
  );
};

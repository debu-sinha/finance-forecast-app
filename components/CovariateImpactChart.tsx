import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { CovariateImpact } from '../types';
import { ArrowUpRight, ArrowDownRight } from 'lucide-react';

interface CovariateImpactChartProps {
  impacts?: CovariateImpact[];
}

export const CovariateImpactChart: React.FC<CovariateImpactChartProps> = ({ impacts }) => {
  if (!impacts || impacts.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-gray-400 bg-gray-50 rounded border border-gray-200 border-dashed">
        <p className="text-sm">No covariates selected for analysis.</p>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex-1 min-h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={impacts}
            margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e5e7eb" />
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis 
              dataKey="name" 
              type="category" 
              tick={{ fontSize: 12, fill: '#4b5563', fontWeight: 500 }} 
              width={100}
            />
            <Tooltip 
              cursor={{ fill: '#f3f4f6' }}
              contentStyle={{ borderRadius: '6px', border: '1px solid #e5e7eb' }}
              formatter={(value: number, name: string, props: any) => {
                 return [`${value.toFixed(1)}%`, 'Impact Score'];
              }}
            />
            <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={20}>
              {impacts.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={entry.direction === 'positive' ? '#1b57b1' : '#dc2626'} 
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Legend / Helper */}
      <div className="mt-4 grid grid-cols-1 gap-2 max-h-32 overflow-y-auto">
         {impacts.map(imp => (
            <div key={imp.name} className="flex items-center justify-between text-xs px-3 py-2 bg-gray-50 rounded border border-gray-100">
               <span className="font-medium text-gray-700">{imp.name}</span>
               <div className="flex items-center space-x-2">
                  <span className={`flex items-center ${imp.direction === 'positive' ? 'text-blue-600' : 'text-red-600'}`}>
                     {imp.direction === 'positive' ? <ArrowUpRight className="w-3 h-3 mr-1"/> : <ArrowDownRight className="w-3 h-3 mr-1"/>}
                     {imp.direction === 'positive' ? 'Positive Corr.' : 'Negative Corr.'}
                  </span>
                  <div className="w-16 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                     <div 
                        className={`h-full rounded-full ${imp.direction === 'positive' ? 'bg-blue-600' : 'bg-red-600'}`} 
                        style={{ width: `${imp.score}%`}}
                     ></div>
                  </div>
                  <span className="font-mono w-8 text-right">{imp.score}%</span>
               </div>
            </div>
         ))}
      </div>
    </div>
  );
};
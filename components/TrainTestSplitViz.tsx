import React from 'react';

interface TrainTestSplitVizProps {
  totalRows: number;
  trainRatio?: number; // Default 0.8
  horizon: number; // Number of future periods
}

export const TrainTestSplitViz: React.FC<TrainTestSplitVizProps> = ({ 
  totalRows, 
  trainRatio = 0.8, 
  horizon 
}) => {
  const trainCount = Math.floor(totalRows * trainRatio);
  const valCount = totalRows - trainCount;
  
  // Calculate widths as percentages
  // We treat totalRows as 100% of the "History" width.
  // The Horizon adds extra width to the right.
  
  // Let's normalize: Total Display Width = History + Horizon
  const totalUnits = totalRows + horizon;
  
  const trainWidth = (trainCount / totalUnits) * 100;
  const valWidth = (valCount / totalUnits) * 100;
  const horizonWidth = (horizon / totalUnits) * 100;

  return (
    <div className="w-full mt-2">
      <div className="flex justify-between text-[10px] text-gray-500 mb-1 font-medium uppercase tracking-wide">
        <span>Timeline Split</span>
        <span>Total: {totalUnits} periods</span>
      </div>
      
      {/* The Bar */}
      <div className="flex w-full h-4 rounded-md overflow-hidden border border-gray-200 shadow-sm">
        
        {/* Training Set */}
        <div 
          style={{ width: `${trainWidth}%` }} 
          className="bg-blue-500 h-full relative group cursor-help"
        >
           <div className="absolute inset-0 flex items-center justify-center text-[9px] text-blue-50 font-bold opacity-0 group-hover:opacity-100 transition-opacity">
             TRAIN ({trainCount})
           </div>
        </div>
        
        {/* Validation Set */}
        <div 
          style={{ width: `${valWidth}%` }} 
          className="bg-orange-400 h-full relative group cursor-help"
        >
            <div className="absolute inset-0 flex items-center justify-center text-[9px] text-orange-50 font-bold opacity-0 group-hover:opacity-100 transition-opacity">
             TEST ({valCount})
           </div>
        </div>
        
        {/* Horizon */}
        <div 
          style={{ width: `${horizonWidth}%` }} 
          className="bg-gray-100 h-full relative group cursor-help"
        >
            {/* Striped Pattern for future */}
            <div className="absolute inset-0 opacity-20" 
                 style={{ backgroundImage: 'linear-gradient(45deg, #000 25%, transparent 25%, transparent 50%, #000 50%, #000 75%, transparent 75%, transparent)', backgroundSize: '8px 8px' }}>
            </div>
            <div className="absolute inset-0 flex items-center justify-center text-[9px] text-gray-500 font-bold opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
             FORECAST (+{horizon})
           </div>
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-start space-x-4 mt-2">
         <div className="flex items-center">
            <div className="w-2 h-2 bg-blue-500 rounded-full mr-1.5"></div>
            <span className="text-[10px] text-gray-600">Training ({Math.round((trainCount/totalRows)*100)}%)</span>
         </div>
         <div className="flex items-center">
            <div className="w-2 h-2 bg-orange-400 rounded-full mr-1.5"></div>
            <span className="text-[10px] text-gray-600">Validation ({Math.round((valCount/totalRows)*100)}%)</span>
         </div>
         <div className="flex items-center">
            <div className="w-2 h-2 bg-gray-200 border border-gray-300 rounded-full mr-1.5"></div>
            <span className="text-[10px] text-gray-600">Future Horizon</span>
         </div>
      </div>
    </div>
  );
};

import React from 'react';

const ConfusionMatrixCell = ({ value, total, isCorrect, rowIndex, colIndex, animateMetrics }) => {
  const intensity = value / total;
  
  return (
    <div
      className={`relative rounded-lg p-4 transition-all duration-500 hover:scale-105 cursor-pointer border-2 ${
        isCorrect 
          ? 'bg-green-50 border-green-200 hover:bg-green-100' 
          : 'bg-red-50 border-red-200 hover:bg-red-100'
      }`}
      style={{ 
        opacity: 0.4 + intensity * 0.6,
        transitionDelay: `${(rowIndex * 3 + colIndex) * 100}ms`
      }}
    >
      <div className={`font-bold text-center text-lg ${isCorrect ? 'text-green-700' : 'text-red-700'}`}>
        {animateMetrics ? value : 0}
      </div>
      <div className="text-center text-xs text-gray-600 mt-1">
        {((value / total) * 100).toFixed(0)}%
      </div>
    </div>
  );
};

export default ConfusionMatrixCell;
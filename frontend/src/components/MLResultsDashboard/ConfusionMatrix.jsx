import React from 'react';
import ConfusionMatrixCell from './ConfusionMatrixCell';

const ConfusionMatrix = ({ matrix, labels, animateMetrics, isVisible }) => (
  <div className={`mb-8 transform transition-all duration-1000 delay-500 ${isVisible ? 'scale-100 opacity-100' : 'scale-95 opacity-0'}`}>
    <div className="bg-white border-2 border-gray-200 rounded-lg p-6 shadow-sm">
      <h2 className="text-2xl font-bold text-gray-900 mb-2">Confusion Matrix</h2>
      <p className="text-gray-600 text-sm mb-6">
        Shows how the model classifies each sample. Rows represent the true classes, and columns represent the model's predictions.  
        Values on the diagonal (green) are correct predictions; all others (red) are classification errors.
      </p>
      <div className="overflow-x-auto">
        <div className="inline-block min-w-full">
          <div className="grid gap-2 mb-4" style={{ gridTemplateColumns: `120px repeat(${labels.length}, 1fr)` }}>
            <div className="flex flex-col items-end justify-end pb-2">
              <span className="text-xs text-gray-500 font-semibold mb-6">Prediction →</span>
              <span className="text-xs text-gray-500 font-semibold mb-2">Real ↓</span>
            </div>
            {labels.map((label, i) => (
              <div key={i} className="text-center text-gray-900 font-semibold text-xs p-2">
                {label}
              </div>
            ))}
            
            {matrix.map((row, i) => (
              <React.Fragment key={i}>
                <div className="text-gray-900 font-semibold text-xs flex items-center justify-end pr-3">
                  {labels[i]}
                </div>
                {row.map((value, j) => {
                  const total = row.reduce((a, b) => a + b, 0);
                  const isCorrect = i === j;
                  
                  return (
                    <ConfusionMatrixCell
                      key={j}
                      value={value}
                      total={total}
                      isCorrect={isCorrect}
                      rowIndex={i}
                      colIndex={j}
                      animateMetrics={animateMetrics}
                    />
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-4 flex justify-center space-x-6 text-sm">
        <div className="flex items-center">
          <div className="w-4 h-4 bg-green-100 border-2 border-green-300 rounded mr-2"></div>
          <span className="text-gray-700">Correct Predictions (diagonal)</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 bg-red-100 border-2 border-red-300 rounded mr-2"></div>
          <span className="text-gray-700">Incorrect Predictions</span>
        </div>
      </div>
    </div>
  </div>
);

export default ConfusionMatrix;
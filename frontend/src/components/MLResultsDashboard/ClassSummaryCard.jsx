import React from 'react';

const ClassSummaryCard = ({ item }) => (
  <div className="bg-white border-2 border-gray-200 rounded-lg p-6 shadow-sm hover:shadow-md transition-shadow">
    <h3 className="text-gray-900 font-bold mb-4 text-lg border-b-2 border-gray-900 pb-2">
      {item.name}
    </h3>
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <span className="text-gray-600 text-sm">Support:</span>
        <span className="text-gray-900 font-bold text-lg">{item.support}</span>
      </div>
      <div className="flex justify-between items-center">
        <span className="text-gray-600 text-sm">Precision:</span>
        <span className="text-gray-900 font-bold">{(item.precision * 100).toFixed(0)}%</span>
      </div>
      <div className="flex justify-between items-center">
        <span className="text-gray-600 text-sm">Recall:</span>
        <span className="text-gray-900 font-bold">{(item.recall * 100).toFixed(0)}%</span>
      </div>
      <div className="flex justify-between items-center pt-2 border-t border-gray-200">
        <span className="text-gray-600 text-sm font-semibold">F1-Score:</span>
        <span className="text-gray-900 font-bold text-lg">{(item.f1 * 100).toFixed(0)}%</span>
      </div>
    </div>
  </div>
);

export default ClassSummaryCard;
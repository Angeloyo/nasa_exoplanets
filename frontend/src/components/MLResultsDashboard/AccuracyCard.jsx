import React from 'react';
import { Award, TrendingUp } from 'lucide-react';

const AccuracyCard = ({ accuracy, animateMetrics, isVisible }) => (
  <div className={`mb-8 transform transition-all duration-1000 delay-200 ${isVisible ? 'scale-100 opacity-100' : 'scale-95 opacity-0'}`}>
    <div className="bg-gray-900 rounded-lg p-8 shadow-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="bg-white p-4 rounded-full">
            <Award className="w-10 h-10 text-gray-900" />
          </div>
          <div>
            <p className="text-gray-400 text-sm uppercase tracking-wide mb-1">Overall Model Accuracy</p>
            <p className="text-5xl font-bold text-white">
              {animateMetrics ? (accuracy * 100).toFixed(2) : '0.00'}%
            </p>
            <p className="text-gray-400 text-sm mt-2">
              The model correctly classifies {(accuracy * 100).toFixed(2)}% of all samples.
            </p>
          </div>
        </div>
        <TrendingUp className="w-20 h-20 text-gray-700" />
      </div>
    </div>
  </div>
);

export default AccuracyCard;
'use client';

import React, { useState, useEffect } from 'react';
import InterpretationGuide from './InterpretationGuide';
import AccuracyCard from './AccuracyCard';
import ConfusionMatrix from './ConfusionMatrix';
import SummaryCardsGrid from './SummaryCardsGrid';
import ROCCurvesSection from './ROCCurvesSection';

const MLResultsDashboard = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [animateMetrics, setAnimateMetrics] = useState(false);

  useEffect(() => {
    setIsVisible(true);
    setTimeout(() => setAnimateMetrics(true), 300);
  }, []);

  const accuracy = 0.8121;
  const confusionMatrix = [
    [736, 239, 39],
    [145, 1075, 92],
    [31, 45, 744]
  ];
  
  const labels = ['FALSE POSITIVE', 'CANDIDATE (PC)', 'CONFIRMED (CP)'];
  
  const classificationData = [
    { name: 'FALSE POSITIVE', precision: 0.81, recall: 0.73, f1: 0.76, support: 1014 },
    { name: 'CANDIDATE (PC)', precision: 0.79, recall: 0.82, f1: 0.80, support: 1312 },
    { name: 'CONFIRMED (CP)', precision: 0.85, recall: 0.91, f1: 0.88, support: 820 }
  ];

  return (
    <div className="min-h-screen bg-white p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className={`text-center mb-12 transform transition-all duration-1000 ${isVisible ? 'translate-y-0 opacity-100' : '-translate-y-10 opacity-0'}`}>
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-2">
            Classification Model Results
          </h1>
        </div>

        <InterpretationGuide isVisible={isVisible} />
        <AccuracyCard accuracy={accuracy} animateMetrics={animateMetrics} isVisible={isVisible} />
        <SummaryCardsGrid data={classificationData} isVisible={isVisible} />
        <ConfusionMatrix 
          matrix={confusionMatrix} 
          labels={labels} 
          animateMetrics={animateMetrics} 
          isVisible={isVisible} 
        />
        
        <ROCCurvesSection isVisible={isVisible} />
      </div>
    </div>
  );
};

export default MLResultsDashboard;
import React from 'react';
import { Info } from 'lucide-react';

const InterpretationGuide = ({ isVisible }) => (
  <div className={`mb-8 transform transition-all duration-1000 delay-100 ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
    <div className="bg-gray-50 border-l-4 border-gray-900 p-6 rounded-r-lg">
      <div className="flex items-start">
        <Info className="w-6 h-6 text-gray-900 mr-3 flex-shrink-0 mt-1" />
        <div>
          <h3 className="font-bold text-gray-900 mb-2">Interpretation Guide</h3>
          <p className="text-gray-700 text-sm leading-relaxed">
            <strong>Accuracy:</strong> The overall percentage of correct predictions made by the model across all samples.
            <br/><br/>
            <strong>Precision:</strong> Of all positive predictions for a class, how many were correct. High precision means few false alarms.
            <br/><br/>
            <strong>Recall (Sensitivity):</strong> Of all actual instances of a class, how many were correctly identified. High recall means the model detects nearly all real cases.
            <br/><br/>
            <strong>F1-Score:</strong> The harmonic mean of precision and recall. It balances both metrics into a single value.
            <br/><br/>
            <strong>Support:</strong> The total number of actual samples for each class in the test dataset.
          </p>
        </div>
      </div>
    </div>
  </div>
);

export default InterpretationGuide;
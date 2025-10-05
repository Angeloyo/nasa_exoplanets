import React from 'react';
import Image from 'next/image'; // Asegúrate de usar Next.js Image si estás en Next.js

const ROCCurvesSection = ({ isVisible }) => (
  <div className={`mb-8 transform transition-all duration-1000 delay-900 ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
    <div className="bg-white border-2 border-gray-200 rounded-lg p-6 shadow-sm">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">ROC Curves</h2>
      <p className="text-gray-700 mb-6">
        Receiver Operating Characteristic (ROC) curves demonstrate our model&apos;s ability to distinguish 
        between classes. The high AUC scores (0.92 for none and candidate, 0.98 for confirmed) indicate 
        excellent classification performance across all categories.
      </p>
      <div className="rounded-lg overflow-hidden border border-gray-300 bg-gray-50 p-4 max-w-2xl mx-auto">
        <Image 
          src="/multiclass_roc_curves.jpeg" 
          alt="Multiclass ROC Curves"
          width={600}
          height={450}
          className="w-full h-auto object-contain"
          priority={false}
        />
      </div>
    </div>
  </div>
);

export default ROCCurvesSection;
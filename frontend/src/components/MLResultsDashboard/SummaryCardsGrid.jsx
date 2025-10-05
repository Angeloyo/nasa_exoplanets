import React from 'react';
import ClassSummaryCard from './ClassSummaryCard';

const SummaryCardsGrid = ({ data, isVisible }) => (
  <div className={`grid grid-cols-1 md:grid-cols-3 gap-6 transform transition-all duration-1000 delay-700 mb-8 ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
    {data.map((item, i) => (
      <ClassSummaryCard key={i} item={item} />
    ))}
  </div>
);

export default SummaryCardsGrid;
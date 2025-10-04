'use client';

import { Upload, Sparkles } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { ConfidenceChart } from '@/components/ConfidenceChart';

// Sample data for demonstration
const sampleResults = [
  { name: 'KOI-123.01', prediction: 'Exoplanet', confidence: 98.5 },
  { name: 'KOI-456.02', prediction: 'Candidate', confidence: 87.3 },
  { name: 'TESS-789.01', prediction: 'Exoplanet', confidence: 95.2 },
  { name: 'K2-234.03', prediction: 'None', confidence: 92.8 },
  { name: 'KOI-567.01', prediction: 'Exoplanet', confidence: 96.7 },
  { name: 'TESS-890.02', prediction: 'Candidate', confidence: 78.4 },
  { name: 'K2-345.01', prediction: 'Exoplanet', confidence: 99.1 },
  { name: 'KOI-678.04', prediction: 'None', confidence: 94.5 },
];

function getPredictionColor(prediction: string) {
  switch (prediction.toLowerCase()) {
    case 'exoplanet':
      return 'text-green-700 bg-green-50';
    case 'candidate':
      return 'text-yellow-700 bg-yellow-50';
    case 'none':
      return 'text-gray-700 bg-gray-50';
    default:
      return 'text-gray-700 bg-gray-50';
  }
}

export default function PredictPage() {
  // Calculate average confidence from sample results
  const averageConfidence = sampleResults.reduce((sum, result) => sum + result.confidence, 0) / sampleResults.length;

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-5xl mx-auto px-6 py-16">
        
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            Exoplanet Prediction
          </h1>
          <p className="text-lg text-gray-600">
            Upload your astronomical data to detect potential exoplanets
          </p>
        </div>

        {/* Upload Section */}
        <div className="mb-12">
          
          {/* CSV Upload Area */}
          <div className="mb-6">
            <label 
              htmlFor="csv-upload"
              className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-gray-300 rounded-xl cursor-pointer hover:border-gray-400 hover:bg-gray-50 transition-all duration-200"
            >
              <div className="flex flex-col items-center justify-center gap-3">
                <Upload className="w-10 h-10 text-gray-400" />
                <div className="text-center">
                  <p className="text-base font-medium text-gray-700">
                    Drop your CSV file here
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    or click to browse
                  </p>
                </div>
              </div>
              <input 
                id="csv-upload" 
                type="file" 
                accept=".csv"
                className="hidden" 
              />
            </label>
          </div>

          {/* Predict Button */}
          <button
            className="w-full bg-gray-900 text-white py-3 px-6 rounded-xl font-semibold hover:bg-gray-800 transition-colors duration-200 flex items-center justify-center gap-3 shadow-sm"
          >
            <Sparkles className="w-5 h-5" />
            Predict Exoplanets
          </button>

        </div>

        {/* Results Section */}
        <div>
          
          {/* Results Header */}
          <div className="mb-6">
            <h2 className="text-2xl font-semibold text-gray-900">
              Prediction Results
            </h2>
            <p className="text-sm text-gray-600 mt-1">
              8 objects analyzed
            </p>
          </div>

          {/* Results Table */}
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="font-semibold text-gray-900">Object Name</TableHead>
                  <TableHead className="font-semibold text-gray-900">Prediction</TableHead>
                  <TableHead className="font-semibold text-gray-900 text-right">Confidence</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sampleResults.map((result, index) => (
                  <TableRow key={index} className="hover:bg-gray-50">
                    <TableCell className="font-medium text-gray-900">
                      {result.name}
                    </TableCell>
                    <TableCell>
                      <span className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${getPredictionColor(result.prediction)}`}>
                        {result.prediction}
                      </span>
                    </TableCell>
                    <TableCell className="text-right text-gray-700 font-medium">
                      {result.confidence}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

        </div>

        {/* Charts Section */}
        <div className="mt-16">
          <ConfidenceChart 
            averageConfidence={averageConfidence}
            totalPredictions={sampleResults.length}
          />
        </div>

      </div>
    </div>
  );
}


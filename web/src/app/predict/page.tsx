'use client';

import { useState } from 'react';
import { Upload, Sparkles, Loader2 } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { ConfidenceChart } from '@/components/ConfidenceChart';
import { PredictionDistributionChart } from '@/components/PredictionDistributionChart';
import { ConfidencePerClassChart } from '@/components/ConfidencePerClassChart';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

function getPredictionColor(prediction: string) {
  switch (prediction.toLowerCase()) {
    case 'exoplanet':
      return 'text-green-700 bg-green-50';
    case 'candidate':
      return 'text-yellow-700 bg-yellow-50';
    case 'none':
      return 'text-red-700 bg-red-50';
    default:
      return 'text-gray-700 bg-gray-50';
  }
}

export default function PredictPage() {
  const [file, setFile] = useState<File | null>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');

  const handlePredict = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (data.status === 'success') {
        setPredictions(data.predictions);
      } else {
        setError(data.error || 'Something went wrong');
      }
    } catch (err) {
      setError('Failed to connect. Make sure the API is running on port 8000');
    } finally {
      setIsLoading(false);
    }
  };

  const averageConfidence = predictions.length > 0
    ? predictions.reduce((sum, r) => sum + r.confidence, 0) / predictions.length
    : 0;

  const exoplanetsCount = predictions.filter(p => p.prediction === 'Exoplanet').length;
  const candidatesCount = predictions.filter(p => p.prediction === 'Candidate').length;
  const falsePositivesCount = predictions.filter(p => p.prediction === 'None').length;

  const filteredPredictions = filter === 'all' 
    ? predictions 
    : predictions.filter(p => p.prediction === filter);

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
                    {file ? file.name : 'Drop your CSV file here'}
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    {file ? 'Click to change file' : 'or click to browse'}
                  </p>
                </div>
              </div>
              <input 
                id="csv-upload" 
                type="file" 
                accept=".csv"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </label>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={!file || isLoading}
            className="w-full bg-gray-900 text-white py-3 px-6 rounded-xl font-semibold hover:bg-gray-800 transition-colors duration-200 flex items-center justify-center gap-3 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Predict Exoplanets
              </>
            )}
          </button>

        </div>

        {/* Results Section */}
        {predictions.length > 0 && (
          <>
            <div>
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-semibold text-gray-900">
                    Prediction Results
                  </h2>
                  <p className="text-sm text-gray-600 mt-1">
                    {predictions.length} objects analyzed
                  </p>
                </div>
                <Select value={filter} onValueChange={setFilter}>
                  <SelectTrigger className="w-48">
                    <SelectValue placeholder="Filter by type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Types</SelectItem>
                    <SelectItem value="Exoplanet">Exoplanet</SelectItem>
                    <SelectItem value="Candidate">Candidate</SelectItem>
                    <SelectItem value="None">False Positive</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="overflow-x-auto max-h-96 overflow-y-auto border border-gray-200 rounded-lg">
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="font-semibold text-gray-900">Object Name</TableHead>
                      <TableHead className="font-semibold text-gray-900">Prediction</TableHead>
                      <TableHead className="font-semibold text-gray-900 text-right">Confidence</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredPredictions.map((result, index) => (
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

            {/* Stats Section */}
            <div className="mt-12 grid grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-green-700">{exoplanetsCount}</div>
                <div className="text-sm text-gray-600 mt-1">Exoplanets Discovered</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-yellow-600">{candidatesCount}</div>
                <div className="text-sm text-gray-600 mt-1">Candidates</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-red-600">{falsePositivesCount}</div>
                <div className="text-sm text-gray-600 mt-1">False Positives</div>
              </div>
            </div>

            {/* Charts Section */}
            <div className="mt-16 grid grid-cols-3 gap-8">
              <ConfidenceChart 
                averageConfidence={averageConfidence}
                totalPredictions={predictions.length}
              />
              <PredictionDistributionChart
                exoplanets={exoplanetsCount}
                candidates={candidatesCount}
                falsePositives={falsePositivesCount}
              />
              <ConfidencePerClassChart predictions={predictions} />
            </div>
          </>
        )}

      </div>
    </div>
  );
}


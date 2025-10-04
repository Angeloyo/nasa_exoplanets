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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import dynamic from 'next/dynamic';

// Dynamically import Planet3D to avoid SSR issues with Three.js
const Planet3D = dynamic(() => import('@/components/Planet3D'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-64 sm:h-80 rounded-xl bg-black flex items-center justify-center">
      <Loader2 className="w-8 h-8 text-white animate-spin" />
    </div>
  ),
});

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

interface Prediction {
  name: string;
  prediction: string;
  confidence: number;
}

export default function PredictPage() {
  const [file, setFile] = useState<File | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [singlePrediction, setSinglePrediction] = useState<Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const [activeTab, setActiveTab] = useState<string>('batch');
  const [singleData, setSingleData] = useState({
    PERIOD: '',
    RADIUS: '',
    DENSITY: '',
    NUM_PLANETS: '',
    DURATION: '',
    TEFF: '',
    DEPTH: ''
  });

  const handleTabChange = (value: string) => {
    setActiveTab(value);
    setError(null);
  };

  const handlePredict = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    const API_URL = process.env.NODE_ENV === 'development' 
      ? 'http://localhost:8000' 
      : 'https://api.exoexplorer.study';

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (data.status === 'success') {
        setPredictions(data.predictions);
      } else {
        setError(data.error || 'Something went wrong');
      }
    } catch {
      setError('Failed to connect. Make sure the API is running on port 8000');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSinglePredict = async () => {
    // Validate all fields are filled
    const allFieldsFilled = Object.values(singleData).every(val => val !== '');
    if (!allFieldsFilled) {
      setError('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    setError(null);

    const API_URL = process.env.NODE_ENV === 'development' 
      ? 'http://localhost:8000' 
      : 'https://api.exoexplorer.study';

    try {
      const requestData = {
        PERIOD: parseFloat(singleData.PERIOD),
        RADIUS: parseFloat(singleData.RADIUS),
        DENSITY: parseFloat(singleData.DENSITY),
        NUM_PLANETS: parseFloat(singleData.NUM_PLANETS),
        DURATION: parseFloat(singleData.DURATION),
        TEFF: parseFloat(singleData.TEFF),
        DEPTH: parseFloat(singleData.DEPTH)
      };

      const res = await fetch(`${API_URL}/predict/single`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      const data = await res.json();

      if (data.status === 'success') {
        setSinglePrediction(data.prediction);
      } else {
        setError(data.error || 'Something went wrong');
      }
    } catch {
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
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-12 md:py-16">
        
        {/* Header */}
        <div className="text-center mb-8 sm:mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-3">
            Exoplanet Prediction
          </h1>
          <p className="text-base sm:text-lg text-gray-600">
            Upload your astronomical data to detect potential exoplanets
          </p>
        </div>

        {/* Prediction Section with Tabs */}
        <div className="mb-8 sm:mb-12">
          <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="batch">Batch Upload</TabsTrigger>
              <TabsTrigger value="single">Single Prediction</TabsTrigger>
            </TabsList>
            
            {/* Batch Upload Tab */}
            <TabsContent value="batch">
              <div className="mb-4 sm:mb-6">
                <label 
                  htmlFor="csv-upload"
                  className="flex flex-col items-center justify-center w-full h-36 sm:h-48 border-2 border-dashed border-gray-300 rounded-xl cursor-pointer hover:border-gray-400 hover:bg-gray-50 transition-all duration-200"
                >
                  <div className="flex flex-col items-center justify-center gap-2 sm:gap-3 px-4">
                    <Upload className="w-8 h-8 sm:w-10 sm:h-10 text-gray-400" />
                    <div className="text-center">
                      <p className="text-sm sm:text-base font-medium text-gray-700">
                        {file ? file.name : 'Drop your CSV file here'}
                      </p>
                      <p className="text-xs sm:text-sm text-gray-500 mt-1">
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

              <button
                onClick={handlePredict}
                disabled={!file || isLoading}
                className="w-full bg-gray-900 text-white py-2.5 sm:py-3 px-4 sm:px-6 rounded-xl text-sm sm:text-base font-semibold hover:bg-gray-800 transition-colors duration-200 flex items-center justify-center gap-2 sm:gap-3 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 sm:w-5 sm:h-5" />
                    Predict Exoplanets
                  </>
                )}
              </button>
            </TabsContent>

            {/* Single Prediction Tab */}
            <TabsContent value="single">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
                <div>
                  <Label htmlFor="period">Period (days)</Label>
                  <Input
                    id="period"
                    type="number"
                    step="any"
                    placeholder="e.g., 6.146653835"
                    value={singleData.PERIOD}
                    onChange={(e) => setSingleData({...singleData, PERIOD: e.target.value})}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="radius">Radius (Earth radii)</Label>
                  <Input
                    id="radius"
                    type="number"
                    step="any"
                    placeholder="e.g., 2.8398685"
                    value={singleData.RADIUS}
                    onChange={(e) => setSingleData({...singleData, RADIUS: e.target.value})}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="density">Density (g/cm³)</Label>
                  <Input
                    id="density"
                    type="number"
                    step="any"
                    placeholder="e.g., 0.993"
                    value={singleData.DENSITY}
                    onChange={(e) => setSingleData({...singleData, DENSITY: e.target.value})}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="num_planets">Number of Planets</Label>
                  <Input
                    id="num_planets"
                    type="number"
                    step="any"
                    placeholder="e.g., 1.0"
                    value={singleData.NUM_PLANETS}
                    onChange={(e) => setSingleData({...singleData, NUM_PLANETS: e.target.value})}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="duration">Duration (hours)</Label>
                  <Input
                    id="duration"
                    type="number"
                    step="any"
                    placeholder="e.g., 1.8522307"
                    value={singleData.DURATION}
                    onChange={(e) => setSingleData({...singleData, DURATION: e.target.value})}
                    className="mt-1"
                  />
                </div>
                <div>
                  <Label htmlFor="teff">Stellar Temperature (K)</Label>
                  <Input
                    id="teff"
                    type="number"
                    step="any"
                    placeholder="e.g., 4968.0"
                    value={singleData.TEFF}
                    onChange={(e) => setSingleData({...singleData, TEFF: e.target.value})}
                    className="mt-1"
                  />
                </div>
                <div className="sm:col-span-2">
                  <Label htmlFor="depth">Transit Depth (ppm)</Label>
                  <Input
                    id="depth"
                    type="number"
                    step="any"
                    placeholder="e.g., 1219.8107094"
                    value={singleData.DEPTH}
                    onChange={(e) => setSingleData({...singleData, DEPTH: e.target.value})}
                    className="mt-1"
                  />
                </div>
              </div>

              <button
                onClick={handleSinglePredict}
                disabled={isLoading}
                className="w-full bg-gray-900 text-white py-2.5 sm:py-3 px-4 sm:px-6 rounded-xl text-sm sm:text-base font-semibold hover:bg-gray-800 transition-colors duration-200 flex items-center justify-center gap-2 sm:gap-3 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 sm:w-5 sm:h-5" />
                    Predict Exoplanet
                  </>
                )}
              </button>
            </TabsContent>
          </Tabs>

          {error && (
            <div className="mt-4 sm:mt-6 p-3 sm:p-4 bg-red-50 border border-red-200 rounded-xl">
              <p className="text-xs sm:text-sm text-red-700">{error}</p>
            </div>
          )}
        </div>

        {/* Batch Results Section */}
        {predictions.length > 0 && activeTab === 'batch' && (
          <>
            <div>
              <div className="mb-4 sm:mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                  <h2 className="text-xl sm:text-2xl font-semibold text-gray-900">
                    Prediction Results
                  </h2>
                  <p className="text-xs sm:text-sm text-gray-600 mt-1">
                    {predictions.length} objects analyzed
                  </p>
                </div>
                <Select value={filter} onValueChange={setFilter}>
                  <SelectTrigger className="w-full sm:w-48">
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

              <div className="overflow-x-auto max-h-[400px] sm:max-h-96 overflow-y-auto border border-gray-200 rounded-lg">
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="font-semibold text-gray-900 text-xs sm:text-sm">Object Name</TableHead>
                      <TableHead className="font-semibold text-gray-900 text-xs sm:text-sm">Prediction</TableHead>
                      <TableHead className="font-semibold text-gray-900 text-right text-xs sm:text-sm">Confidence</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredPredictions.map((result, index) => (
                      <TableRow key={index} className="hover:bg-gray-50">
                        <TableCell className="font-medium text-gray-900 text-xs sm:text-sm">
                          {result.name}
                        </TableCell>
                        <TableCell>
                          <span className={`inline-flex px-2 sm:px-3 py-0.5 sm:py-1 rounded-full text-xs sm:text-sm font-medium ${getPredictionColor(result.prediction)}`}>
                            {result.prediction}
                          </span>
                        </TableCell>
                        <TableCell className="text-right text-gray-700 font-medium text-xs sm:text-sm">
                          {result.confidence}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>

            {/* Stats Section */}
            <div className="mt-10 sm:mt-16 grid grid-cols-1 sm:grid-cols-3 gap-6 sm:gap-12">
              <div className="text-center">
                <div className="text-4xl sm:text-6xl font-extrabold text-green-700">{exoplanetsCount}</div>
                <div className="text-base sm:text-lg text-gray-600 mt-2 font-medium">Exoplanets Discovered</div>
              </div>
              <div className="text-center">
                <div className="text-4xl sm:text-6xl font-extrabold text-yellow-600">{candidatesCount}</div>
                <div className="text-base sm:text-lg text-gray-600 mt-2 font-medium">Candidates</div>
              </div>
              <div className="text-center">
                <div className="text-4xl sm:text-6xl font-extrabold text-red-600">{falsePositivesCount}</div>
                <div className="text-base sm:text-lg text-gray-600 mt-2 font-medium">False Positives</div>
              </div>
            </div>

            {/* Charts Section */}
            <div className="mt-12 sm:mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8">
              <PredictionDistributionChart
                exoplanets={exoplanetsCount}
                candidates={candidatesCount}
                falsePositives={falsePositivesCount}
              />
              <ConfidenceChart 
                averageConfidence={averageConfidence}
                totalPredictions={predictions.length}
              />
              <ConfidencePerClassChart predictions={predictions} />
            </div>
          </>
        )}

        {/* Single Prediction Results Section */}
        {singlePrediction && activeTab === 'single' && (
          <div className="mt-8">
            <h2 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-6">
              Prediction Result
            </h2>
            
            <div className="flex flex-col items-center gap-6">
              <div className="text-center">
                <h3 className="text-lg sm:text-3xl font-semibold ">
                  Object: {singlePrediction.name}
                </h3>
              </div>

              {/* 3D Planet Visualization */}
              <div className="w-full max-w-2xl">
                <Planet3D
                  temperature={parseFloat(singleData.TEFF)}
                  radius={parseFloat(singleData.RADIUS)}
                  density={parseFloat(singleData.DENSITY)}
                />
              </div>

              {/* Main Prediction Card */}
              <div className={`w-full max-w-md rounded-xl p-6 sm:p-8 text-center shadow-md ${
                singlePrediction.prediction === 'Exoplanet' 
                  ? 'bg-green-50 border-2 border-green-300' 
                  : singlePrediction.prediction === 'Candidate'
                  ? 'bg-yellow-50 border-2 border-yellow-300'
                  : 'bg-red-50 border-2 border-red-300'
              }`}>
                <p className="text-sm sm:text-base font-medium text-gray-600 mb-2">
                  Classification
                </p>
                <p className={`text-3xl sm:text-4xl font-bold mb-4 ${
                  singlePrediction.prediction === 'Exoplanet' 
                    ? 'text-green-700' 
                    : singlePrediction.prediction === 'Candidate'
                    ? 'text-yellow-700'
                    : 'text-red-700'
                }`}>
                  {singlePrediction.prediction}
                </p>
                <div className="pt-4 border-t border-gray-300">
                  <p className="text-sm sm:text-base font-medium text-gray-600 mb-1">
                    Confidence Level
                  </p>
                  <p className="text-2xl sm:text-3xl font-bold text-gray-900">
                    {singlePrediction.confidence}%
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}


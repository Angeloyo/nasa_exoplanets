'use client';

import Image from 'next/image';
import MLResultsDashboard from '@/components/MLResultsDashboard';
import { 
  Accordion, 
  AccordionItem, 
  AccordionTrigger, 
  AccordionContent 
} from '@/components/ui/accordion';

export default function LearnPage() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        {/* Header */}
        <div className="mb-8 sm:mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-3 sm:mb-4">
            How Scientists Discover Exoplanets
          </h1>
          <p className="text-lg sm:text-xl text-gray-600">
            From space telescopes to machine learning: A step-by-step guide to finding planets beyond our solar system
          </p>
        </div>

        {/* 4-Step Process */}
        <div className="mb-12">
          <Accordion type="single" collapsible className="space-y-4">
            
            {/* Step 1: Data Collection from Missions */}
            <AccordionItem value="step-1" className="border-2 border-gray-200 rounded-lg px-6">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 hover:no-underline">
                <span className="flex items-center gap-3">
                  <span className="flex-shrink-0 w-8 h-8 bg-gray-900 text-white rounded-full flex items-center justify-center text-sm font-bold">1</span>
                  Gather Data from Space Missions
                </span>
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 space-y-6">
                <p className="text-base leading-relaxed">
                  The first step in discovering exoplanets is collecting data from NASA&apos;s space telescopes. 
                  These missions continuously monitor thousands of stars, recording tiny changes in their brightness 
                  over time. This data becomes the foundation for all exoplanet detection.
                </p>

                {/* Kepler */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    Kepler Mission (2009-2018)
                  </h3>
                  <p className="mb-3 leading-relaxed">
                    Kepler was NASA&apos;s first mission dedicated to finding Earth-sized planets. It stared at a 
                    single patch of sky containing over 150,000 stars for nearly four years, discovering over 
                    2,600 confirmed exoplanets.
                  </p>
                  <div className="rounded-lg overflow-hidden">
                    <Image 
                      src="/kepler.jpg" 
                      alt="Kepler Space Telescope"
                      width={800}
                      height={450}
                      className="w-full h-auto"
                    />
                  </div>
                </div>

                {/* K2 */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    K2 Mission (2014-2018)
                  </h3>
                  <p className="leading-relaxed">
                    After Kepler&apos;s reaction wheels failed, NASA repurposed the spacecraft for the K2 mission. 
                    K2 observed different fields along the ecliptic plane through multiple campaigns, successfully 
                    discovering over 500 confirmed exoplanets despite hardware limitations.
                  </p>
                </div>

                {/* TESS */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    TESS Mission (2018-Present)
                  </h3>
                  <p className="mb-3 leading-relaxed">
                    The Transiting Exoplanet Survey Satellite (TESS) is NASA&apos;s current planet-hunting mission. 
                    Unlike Kepler, TESS surveys the entire sky, scanning 85% of it and focusing on nearby bright stars. 
                    It has discovered over 400 confirmed exoplanets and continues finding more.
                  </p>
                  <div className="rounded-lg overflow-hidden">
                    <Image 
                      src="/tess.webp" 
                      alt="TESS Space Telescope"
                      width={800}
                      height={450}
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* Step 2: Transit Method Detection */}
            <AccordionItem value="step-2" className="border-2 border-gray-200 rounded-lg px-6">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 hover:no-underline">
                <span className="flex items-center gap-3">
                  <span className="flex-shrink-0 w-8 h-8 bg-gray-900 text-white rounded-full flex items-center justify-center text-sm font-bold">2</span>
                  Detect Transits and Extract Features
                </span>
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 space-y-4">
                <p className="text-base leading-relaxed">
                  The transit method detects exoplanets by observing the tiny decrease in a star&apos;s brightness 
                  when a planet passes in front of it. When a planet transits its host star, it blocks a small 
                  fraction of light, creating a characteristic dip in the light curve.
                </p>
                
                <p className="leading-relaxed">
                  Scientists analyze these transits to extract key features and gather as much information as possible 
                  about the potential planet:
                </p>

                <ul className="list-disc list-inside space-y-2 ml-4">
                  <li><strong>Orbital Period:</strong> Time it takes for a planet to complete one orbit around its star</li>
                  <li><strong>Transit Duration:</strong> How long the planet blocks the star&apos;s light during each pass</li>
                  <li><strong>Transit Depth:</strong> How much the star&apos;s brightness decreases during transit</li>
                  <li><strong>Planetary Radius:</strong> Size of the planet relative to its host star</li>
                  <li><strong>Stellar Parameters:</strong> Properties of the host star (temperature, size, mass)</li>
                  <li><strong>Signal-to-Noise Ratio:</strong> Quality and reliability of the detection signal</li>
                </ul>

                <p className="leading-relaxed">
                  These brightness dips are extremely small—sometimes less than 1% of the star&apos;s total brightness—which 
                  is why sensitive space telescopes are essential for detection.
                </p>

                {/* Video Embed */}
                <div className="rounded-lg overflow-hidden shadow-lg mt-4">
                  <div className="relative w-full" style={{ paddingBottom: '56.25%' }}>
                    <iframe
                      className="absolute top-0 left-0 w-full h-full"
                      src="https://www.youtube.com/embed/bv2BV82J0Jk"
                      title="Transit Method Explained"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                    />
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* Step 3: Manual Classification */}
            <AccordionItem value="step-3" className="border-2 border-gray-200 rounded-lg px-6">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 hover:no-underline">
                <span className="flex items-center gap-3">
                  <span className="flex-shrink-0 w-8 h-8 bg-gray-900 text-white rounded-full flex items-center justify-center text-sm font-bold">3</span>
                  Traditional Manual Classification
                </span>
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 space-y-4">
                <p className="text-base leading-relaxed">
                  Traditionally, astronomers manually analyze light curves and extracted features to classify each 
                  detection into one of three categories:
                </p>

                <div className="space-y-4 mt-4">
                  <div className="border-l-4 border-green-500 bg-green-50 p-4 rounded-r-lg">
                    <h4 className="font-bold text-green-900 mb-1">Confirmed Exoplanet</h4>
                    <p className="text-sm text-green-800">
                      High-confidence detections with clear, repeating transit signals and confirmed through 
                      follow-up observations. These are verified planets orbiting distant stars.
                    </p>
                  </div>

                  <div className="border-l-4 border-yellow-500 bg-yellow-50 p-4 rounded-r-lg">
                    <h4 className="font-bold text-yellow-900 mb-1">Planetary Candidate</h4>
                    <p className="text-sm text-yellow-800">
                      Promising detections that show planet-like signals but require additional observations 
                      or analysis to rule out false positives and confirm their planetary nature.
                    </p>
                  </div>

                  <div className="border-l-4 border-red-500 bg-red-50 p-4 rounded-r-lg">
                    <h4 className="font-bold text-red-900 mb-1">False Positive</h4>
                    <p className="text-sm text-red-800">
                      Signals caused by other phenomena like eclipsing binary stars, stellar activity, 
                      or instrumental artifacts that mimic planet transits but aren&apos;t actual planets.
                    </p>
                  </div>
                </div>

                <div className="bg-gray-50 border border-gray-200 p-4 rounded-lg mt-6">
                  <p className="text-sm leading-relaxed">
                    <strong>The Challenge:</strong> With missions like TESS generating enormous amounts of data, 
                    manual classification becomes extremely time-consuming. Analyzing thousands of light curves 
                    by hand can take months or even years, creating a bottleneck in the discovery process.
                  </p>
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* Step 4: Our ML Approach */}
            <AccordionItem value="step-4" className="border-2 border-gray-200 rounded-lg px-6 !border-b-2">
              <AccordionTrigger className="text-xl font-semibold text-gray-900 hover:no-underline">
                <span className="flex items-center gap-3">
                  <span className="flex-shrink-0 w-8 h-8 bg-gray-900 text-white rounded-full flex items-center justify-center text-sm font-bold">4</span>
                  Our Approach: Machine Learning Automation
                </span>
              </AccordionTrigger>
              <AccordionContent className="text-gray-700 space-y-4">
                <p className="text-base leading-relaxed">
                  Instead of manual classification, we use machine learning to automate the detection process. 
                  Our approach leverages data that NASA researchers have already classified to train intelligent 
                  models that can predict new detections.
                </p>

                <div className="bg-blue-50 border-l-4 border-blue-500 p-5 rounded-r-lg">
                  <h4 className="font-bold text-blue-900 mb-2">Training Dataset</h4>
                  <p className="text-sm text-blue-800 leading-relaxed">
                    We trained our models on <strong>over 20,000 observations</strong> from Kepler, K2, and TESS 
                    missions that have been manually classified by NASA researchers. This dataset includes confirmed 
                    exoplanets, planetary candidates, and false positives—giving our model a comprehensive understanding 
                    of what each category looks like.
                  </p>
                </div>

                <div className="space-y-3">
                  <h4 className="font-semibold text-gray-900">Model Selection</h4>
                  <p className="leading-relaxed">
                    We experimented with multiple machine learning approaches including neural networks, 
                    random forests, and gradient boosting algorithms. After extensive testing and hyperparameter 
                    tuning, we found that <strong>XGBoost</strong> delivered the best performance.
                  </p>
                </div>

                {/* <div className="bg-green-50 border border-green-200 p-5 rounded-lg">
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 w-12 h-12 bg-green-600 text-white rounded-full flex items-center justify-center text-xl font-bold">
                      81%
                    </div>
                    <div>
                      <h4 className="font-bold text-green-900 mb-1">Impressive Accuracy</h4>
                      <p className="text-sm text-green-800 leading-relaxed">
                        Our XGBoost model achieves <strong>81% accuracy</strong> in classifying observations 
                        into confirmed exoplanets, candidates, or false positives—processing in seconds what 
                        would take humans months to analyze.
                      </p>
                    </div>
                  </div>
                </div> */}

                <div className="space-y-3 mt-6">
                  <h4 className="font-semibold text-gray-900">Why This Matters</h4>
                  <p className="leading-relaxed">
                    Automated machine learning classification enables scientists to:
                  </p>
                  <ul className="list-disc list-inside space-y-2 ml-4">
                    <li>Process thousands of observations in seconds instead of months</li>
                    <li>Identify subtle patterns that might be missed by human analysis</li>
                    <li>Consistently apply classification criteria across entire datasets</li>
                    <li>Prioritize the most promising candidates for follow-up observations</li>
                    <li>Discover rare or unusual planetary systems</li>
                  </ul>
                </div>

                <div className="bg-gray-900 text-white p-5 rounded-lg mt-6">
                  <p className="text-sm leading-relaxed">
                    <strong>Try it yourself!</strong> Visit our <a href="/discover" className="underline font-semibold">Discover</a> page 
                    to use sample data and see our machine learning model classify potential exoplanets in real-time.
                  </p>
                </div>

                {/* ML Results Dashboard */}
                <div className="mt-8 pt-8 border-t-2 border-gray-200 pb-6">
                  <div className="mb-6">
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      Model Performance Metrics
                    </h3>
                    <p className="text-sm text-gray-600">
                      Detailed evaluation results from our XGBoost classification model
                    </p>
                  </div>
                  <MLResultsDashboard />
                </div>
              </AccordionContent>
            </AccordionItem>

          </Accordion>
        </div>

      </div>
    </div>
  );
}

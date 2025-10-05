import Image from 'next/image';
import MLResultsDashboard from '@/components/MLResultsDashboard';

export default function LearnPage() {
  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 py-8 sm:py-12">
        {/* Header */}
        <div className="mb-8 sm:mb-12">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-3 sm:mb-4">
            Understanding Exoplanet Detection
          </h1>
          <p className="text-lg sm:text-xl text-gray-600">
            Learn how scientists discover planets beyond our solar system using space telescopes and machine learning
          </p>
        </div>

        {/* Section 1: Transit Method */}
        <section className="mb-12 sm:mb-16">
          <h2 className="text-2xl sm:text-3xl font-semibold text-gray-900 mb-3 sm:mb-4">
            The Transit Method
          </h2>

          <div className="prose prose-base sm:prose-lg max-w-none text-gray-700 space-y-3 sm:space-y-4 mb-6">
            <p>
              The transit method is the most successful technique for finding exoplanets. It works by detecting 
              the tiny decrease in a star&apos;s brightness when a planet passes in front of it from our point of view.
            </p>
            <p>
              When a planet transits its host star, it blocks a small fraction of the star&apos;s light, creating 
              a characteristic dip in the light curve. By repeatedly observing these transits, scientists can 
              confirm the presence of a planet and determine key properties like its size and orbital period.
            </p>
            <p>
              These brightness dips are extremely small, which is why sensitive space 
              telescopes are needed to detect them.
            </p>
          </div>
          
          {/* Video Embed */}
          <div className="rounded-lg overflow-hidden shadow-lg">
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
        </section>

        {/* Section 2: Space Missions */}
        <section className="mb-12 sm:mb-16">
          <h2 className="text-2xl sm:text-3xl font-semibold text-gray-900 mb-4 sm:mb-6">
            NASA&apos;s Exoplanet Missions
          </h2>
          
          {/* Kepler */}
          <div className="mb-8 sm:mb-12">
            <h3 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-3">
              Kepler Mission (2009-2018)
            </h3>

            <div className="text-gray-700 space-y-3 mb-4">
              <p>
                Kepler was NASA&apos;s first mission dedicated to finding Earth-sized planets in the habitable zones 
                of distant stars. It stared at a single patch of sky containing over 150,000 stars for nearly 
                four years. The mission discovered over 2,600 confirmed exoplanets and proved that planets are 
                common in our galaxy.
              </p>
            </div>
            
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
          <div className="mb-8 sm:mb-12">
            <h3 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-3">
              K2 Mission (2014-2018)
            </h3>
            <div className="text-gray-700 space-y-3 mb-4">
              <p>
                After Kepler&apos;s reaction wheels failed, NASA repurposed the spacecraft for the K2 mission. 
                Instead of staring at one field, K2 observed different fields of view along the ecliptic plane 
                through multiple observation campaigns. The mission successfully discovered over 500 confirmed 
                exoplanets, extending Kepler&apos;s legacy despite hardware limitations.
              </p>
            </div>
          </div>

          {/* TESS */}
          <div className="mb-8 sm:mb-12">
            <h3 className="text-xl sm:text-2xl font-semibold text-gray-900 mb-3">
              TESS Mission (2018-Present)
            </h3>

            <div className="text-gray-700 space-y-3 mb-4">
              <p>
                The Transiting Exoplanet Survey Satellite (TESS) is NASA&apos;s current planet-hunting mission. 
                Unlike Kepler, TESS surveys the entire sky, scanning 85% of it and focusing on nearby bright stars 
                for follow-up studies. Since its launch, the mission has discovered over 400 confirmed exoplanets 
                and continues to find more.
              </p>
            </div>
            
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
        </section>

        {/* Section 3: Machine Learning */}
        <section className="mb-12 sm:mb-16">
          <h2 className="text-2xl sm:text-3xl font-semibold text-gray-900 mb-4 sm:mb-6">
            Machine Learning for Exoplanet Detection
          </h2>
          <div className="prose prose-base sm:prose-lg max-w-none text-gray-700 space-y-3 sm:space-y-4 mb-6">
            <p>
              Traditionally, astronomers manually analyzed light curves to identify potential exoplanets—a 
              time-consuming process given the massive datasets from space missions. Machine learning offers 
              an automated solution.
            </p>
            <p>
              AI models are trained on datasets containing confirmed exoplanets, planetary candidates, and 
              false positives. The models learn to recognize patterns in various features:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li><strong>Orbital Period:</strong> Time it takes for a planet to complete one orbit</li>
              <li><strong>Transit Duration:</strong> How long the planet blocks the star&apos;s light</li>
              <li><strong>Transit Depth:</strong> How much the star&apos;s brightness decreases</li>
              <li><strong>Planetary Radius:</strong> Size of the planet relative to its star</li>
              <li><strong>Stellar Parameters:</strong> Properties of the host star</li>
            </ul>
            <p>
              By analyzing these features, machine learning models can quickly classify new observations 
              as confirmed exoplanets, planetary candidates, or false positives with high accuracy. This 
              automation allows scientists to process vast amounts of data and potentially discover planets 
              that might have been missed by manual analysis.
            </p>
          </div>
        </section>

        {/* Section 4: Our ML Approach */}
        <section className="mb-12 sm:mb-16">
          <h2 className="text-2xl sm:text-3xl font-semibold text-gray-900 mb-4 sm:mb-6">
            Our Machine Learning Approach
          </h2>
          <div className="prose prose-base sm:prose-lg max-w-none text-gray-700 space-y-3 sm:space-y-4 mb-6">
            <p>
              For this project, we trained and evaluated multiple machine learning models including neural networks, 
              random forest classifiers, and gradient boosting algorithms on data from Kepler, K2, and TESS missions.
            </p>
            <p>
              After extensive experimentation and hyperparameter tuning, we found that <strong>XGBoost</strong> delivered 
              the best performance with an impressive <strong>81% accuracy</strong> in classifying exoplanet observations 
              into three categories: confirmed exoplanets, planetary candidates, and false positives.
            </p>
          </div>
        </section>

        {/* ML Results Dashboard */}
        <MLResultsDashboard />

        {/* Section 5: Why This Matters */}
        <section className="mb-12 sm:mb-16">
          <h2 className="text-2xl sm:text-3xl font-semibold text-gray-900 mb-4 sm:mb-6">
            Why Automated Detection Matters
          </h2>
          <div className="prose prose-base sm:prose-lg max-w-none text-gray-700 space-y-3 sm:space-y-4">
            <p>
              With missions like TESS generating enormous amounts of data, automated classification becomes 
              essential. Machine learning can:
            </p>
            <ul className="list-disc list-inside space-y-2 ml-4">
              <li>Process thousands of light curves in seconds instead of months</li>
              <li>Identify subtle patterns that might be missed by human analysis</li>
              <li>Consistently apply classification criteria across entire datasets</li>
              <li>Help prioritize targets for follow-up observations</li>
              <li>Discover rare or unusual planetary systems</li>
            </ul>
            <p>
              Our XGBoost model, trained on publicly available data from Kepler, K2, and TESS, demonstrates 
              that machine learning can effectively automate exoplanet detection, helping advance our understanding 
              of planets beyond our solar system.
            </p>
          </div>
        </section>


      </div>
    </div>
  );
}


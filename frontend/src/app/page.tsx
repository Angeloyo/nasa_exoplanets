import Galaxy from '@/components/Galaxy';
import { Sparkles } from 'lucide-react';

export default function Home() {
  return (
    <div className="min-h-[92vh] flex flex-col relative bg-black">
      {/* Galaxy Background */}
      <div className="absolute inset-0 w-full h-full pointer-events-auto">
        <Galaxy 
          mouseRepulsion={false}
          mouseInteraction={true}
          density={1}
          glowIntensity={0.2}
          saturation={0}
          hueShift={140}
          twinkleIntensity={0.3}
          rotationSpeed={0.05}
          repulsionStrength={2}
          autoCenterRepulsion={0}
          starSpeed={0.1}
          speed={0.1}
          transparent={false}
        />
      </div>

      {/* Content */}
      <main className="flex-1 flex items-center justify-center px-4 sm:px-6 relative z-10 pointer-events-none">
        <div className="max-w-4xl text-center space-y-6 sm:space-y-8">
          <div className="space-y-3 sm:space-y-4">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white">
              Discover Exoplanets with AI
            </h2>
            <p className="text-base sm:text-lg md:text-xl text-gray-300 px-4">
              Automated exoplanet detection using machine learning on NASA&apos;s Kepler, K2, and TESS datasets
            </p>
          </div>

          <div className="pointer-events-auto">
            <a
              href="/predict"
              className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-black bg-white hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-white transition-colors duration-200"
            >
              <Sparkles className="mr-2 h-5 w-5" />
              Predict Exoplanets
            </a>
          </div>

        </div>
      </main>

    </div>
  );
}

import Galaxy from '@/components/Galaxy';

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
      <main className="flex-1 flex items-center justify-center px-6 relative z-10 pointer-events-none">
        <div className="max-w-4xl text-center space-y-8">
          <div className="space-y-4">
            <h2 className="text-5xl font-bold tracking-tight text-white">
              Discover Exoplanets with AI
            </h2>
            <p className="text-xl text-gray-300">
              Automated exoplanet detection using machine learning on NASA's Kepler, K2, and TESS datasets
            </p>
          </div>

        </div>
      </main>

    </div>
  );
}

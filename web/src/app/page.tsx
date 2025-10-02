export default function Home() {
  return (
    <div className="min-h-[92vh] flex flex-col">
      <main className="flex-1 flex items-center justify-center px-6">
        <div className="max-w-3xl text-center space-y-8">
          <div className="space-y-4">
            <h2 className="text-5xl font-bold tracking-tight">
              Discover Exoplanets with AI
            </h2>
            <p className="text-xl text-muted-foreground">
              Automated exoplanet detection using machine learning on NASA's Kepler, K2, and TESS datasets
            </p>
          </div>

        </div>
      </main>

    </div>
  );
}

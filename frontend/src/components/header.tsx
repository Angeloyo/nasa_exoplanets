import Link from "next/link";
import Image from "next/image";

export function Header() {
  return (
    <header className="border-b">
      <div className="container mx-auto px-6 py-5 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-3 text-2xl font-semibold hover:opacity-80 transition-opacity">
          <Image 
            src="/NASA_logo.svg" 
            alt="NASA Logo" 
            width={55} 
            height={46}
            className="h-12 w-auto"
          />
          ExoExplorer
        </Link>
        
        <nav className="flex gap-8">
          <Link 
            href="/predict" 
            className="text-base font-medium hover:text-primary transition-colors"
          >
            Predict
          </Link>
          <Link 
            href="/learn" 
            className="text-base font-medium hover:text-primary transition-colors"
          >
            Learn
          </Link>
          <Link 
            href="/about" 
            className="text-base font-medium hover:text-primary transition-colors"
          >
            About
          </Link>
        </nav>
      </div>
    </header>
  );
}


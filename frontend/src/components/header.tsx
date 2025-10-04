'use client';

import Link from "next/link";
import Image from "next/image";
import { useState } from "react";
import { Menu, X } from "lucide-react";

export function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="border-b bg-white relative z-50">
      <div className="container mx-auto px-4 sm:px-6 py-4 sm:py-5 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2 sm:gap-3 text-xl sm:text-2xl font-semibold hover:opacity-80 transition-opacity">
          <Image 
            src="/NASA_logo.svg" 
            alt="NASA Logo" 
            width={55} 
            height={46}
            className="h-8 sm:h-12 w-auto"
          />
          <span>ExoExplorer</span>
        </Link>
        
        {/* Desktop Navigation */}
        <nav className="hidden md:flex gap-6 lg:gap-8">
          <Link 
            href="/discover" 
            className="text-base font-medium hover:text-primary transition-colors"
          >
            Discover
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

        {/* Mobile Menu Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="md:hidden p-2 hover:bg-gray-100 rounded-lg transition-colors"
          aria-label="Toggle menu"
        >
          {mobileMenuOpen ? (
            <X className="h-6 w-6" />
          ) : (
            <Menu className="h-6 w-6" />
          )}
        </button>
      </div>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <nav className="md:hidden border-t bg-white">
          <div className="container mx-auto px-4 py-4 flex flex-col gap-4">
            <Link 
              href="/discover" 
              className="text-base font-medium hover:text-primary transition-colors py-2"
              onClick={() => setMobileMenuOpen(false)}
            >
              Discover
            </Link>
            <Link 
              href="/learn" 
              className="text-base font-medium hover:text-primary transition-colors py-2"
              onClick={() => setMobileMenuOpen(false)}
            >
              Learn
            </Link>
            <Link 
              href="/about" 
              className="text-base font-medium hover:text-primary transition-colors py-2"
              onClick={() => setMobileMenuOpen(false)}
            >
              About
            </Link>
          </div>
        </nav>
      )}
    </header>
  );
}


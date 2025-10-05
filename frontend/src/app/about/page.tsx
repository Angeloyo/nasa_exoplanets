export default function AboutPage() {
  const team = [
    { 
      name: 'Ángel Sanchez Guerrero', 
      image: '/angel.jpeg', 
      imageOg: '/angelOg.jpg', 
      rotation: '-3deg', 
      role: 'Frontend Development & UI/UX Design',
      github: 'https://github.com/Angeloyo'
    },
    { 
      name: 'Raúl Martínez Alonso', 
      image: '/raul.jpeg', 
      imageOg: '/raulOg.jpg', 
      rotation: '2deg', 
      role: 'Data Science & documentation',
      github: 'https://github.com/raulmart03'
    },
    { 
      name: 'Javier Trujillo Castro', 
      image: '/javi.jpeg', 
      imageOg: '/javiOg.jpg', 
      rotation: '-2deg', 
      role: 'Machine Learning & content creator',
      github: 'https://github.com/javitrucas'
    },
    { 
      name: 'Pablo Tamayo López', 
      image: '/pablo.jpeg', 
      imageOg: '/pabloOg.jpg', 
      rotation: '3deg', 
      role: 'Backend Development & content creator',
      github: 'https://github.com/pablotl0'
    },
  ];

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-12 sm:py-16">
        
        {/* Header */}
        <div className="text-center mb-12 sm:mb-16">
          <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-3">
            About Us
          </h1>
          <p className="text-base sm:text-lg text-gray-600">
            Meet the team behind Exo Explorer
          </p>
        </div>

        {/* Team Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 sm:gap-10 lg:gap-12">
          {team.map((member, index) => (
            <div key={index} className="flex flex-col items-center">
              <a
                href={member.github}
                target="_blank"
                rel="noopener noreferrer"
                className="focus:outline-none"
                tabIndex={0}
                aria-label={`Open ${member.name}'s GitHub in a new tab`}
              >
                <div 
                  className="relative bg-white p-3 sm:p-4 shadow-lg border-4 border-gray-800 hover:scale-105 transition-transform duration-300 group"
                  style={{ 
                    transform: `rotate(${member.rotation})`,
                    width: '160px',
                    height: '192px'
                  }}
                >
                  <div className="w-full h-full bg-gray-200 relative overflow-hidden">
                    <img
                      src={member.image}
                      alt={member.name}
                      className="w-full h-full object-cover transition-opacity duration-300 group-hover:opacity-0"
                    />
                    <img
                      src={member.imageOg}
                      alt={`${member.name} original`}
                      className="w-full h-full object-cover absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
                    />
                  </div>
                </div>
              </a>
              <div className="mt-4 sm:mt-6 text-center px-2">
                <p className="text-lg sm:text-xl font-semibold text-gray-900">
                  {member.name}
                </p>
                <p className="text-xs sm:text-sm text-gray-600 mt-1">
                  {member.role}
                </p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-16 text-center">
          <span className="italic text-gray-500 text-sm">
            We swear the drawings aren&apos;t AI-generated, Javi made them!
          </span>
        </div>
      </div>
    </div>
  );
}
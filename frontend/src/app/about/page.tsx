export default function AboutPage() {
  const team = [
    { name: 'Ángel Sanchez Guerrero', image: '/angel.jpeg', rotation: '-3deg', role: 'Frontend Development & UI/UX Design' },
    { name: 'Raúl Martínez Alonso', image: '/raul.jpeg', rotation: '2deg', role: 'Data Science & documentation' },
    { name: 'Javier Trujillo Castro', image: '/javi.jpeg', rotation: '-2deg', role: 'Machine Learning & content creator' },
    { name: 'Pablo Tamayo López', image: '/pablo.jpeg', rotation: '3deg', role: 'Backend Development & content creator' },
  ];

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-6 py-16">
        
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            About Us
          </h1>
          <p className="text-lg text-gray-600">
            Meet the team behind Exo Explorer
          </p>
        </div>

        {/* Team Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-12">
          {team.map((member, index) => (
            <div key={index} className="flex flex-col items-center">
              <div 
                className="relative bg-white p-4 shadow-lg border-4 border-gray-800 hover:scale-105 transition-transform duration-300"
                style={{ 
                  transform: `rotate(${member.rotation})`,
                  width: '200px',
                  height: '240px'
                }}
              >
                <div className="w-full h-full bg-gray-200 relative overflow-hidden">
                  <img
                    src={member.image}
                    alt={member.name}
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
              <div className="mt-6 text-center">
                <p className="text-xl font-semibold text-gray-900">
                  {member.name}
                </p>
                <p className="text-sm text-gray-600 mt-1">
                  {member.role}
                </p>
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}


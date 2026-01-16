import React, { useState } from 'react';
import { Upload, Search, Filter, Download, User, Briefcase, Award, MapPin, TrendingUp, AlertCircle, CheckCircle, XCircle, Loader } from 'lucide-react';

const ResumeScreeningSystem = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [candidates, setCandidates] = useState([]);
  const [jobDescription, setJobDescription] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [filters, setFilters] = useState({ minScore: 0, skills: '', location: '' });
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState('');
  const [jobTitle, setJobTitle] = useState('');

  const API_URL = 'http://localhost:5000';

  const demoJobDescription = `Senior Full Stack Developer

Required Skills:
- Python, Django/Flask (5+ years)
- React.js, TypeScript (3+ years)
- PostgreSQL, MongoDB
- RESTful API design
- AWS/Azure cloud platforms
- Docker, Kubernetes
- Git version control

Preferred:
- Machine Learning experience
- CI/CD pipeline setup
- Team leadership experience

Education: Bachelor's in Computer Science or related field
Experience: 5+ years in software development`;

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    setSelectedFiles(files);
    setError('');
  };

  const handleFileDrop = (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => 
      file.name.endsWith('.pdf') || file.name.endsWith('.docx') || file.name.endsWith('.doc')
    );
    setSelectedFiles(validFiles);
    setError('');
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleProcessResumes = async () => {
    // Validation
    if (!jobDescription.trim()) {
      setError('Please provide a job description');
      return;
    }

    if (selectedFiles.length === 0) {
      setError('Please upload at least one resume');
      return;
    }

    setProcessing(true);
    setError('');

    try {
      // Create FormData object
      const formData = new FormData();
      formData.append('jobDescription', jobDescription);
      
      // Append all resume files
      selectedFiles.forEach((file) => {
        formData.append('resumes', file);
      });

      // Make API request
      const response = await fetch(`${API_URL}/api/screen`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - browser will set it with boundary
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Processing failed');
      }

      const data = await response.json();
      
      // Update state with results
      setCandidates(data.candidates || []);
      setJobTitle(data.job_title || 'Position');
      setActiveTab('dashboard');

    } catch (err) {
      console.error('Processing error:', err);
      setError(err.message || 'Failed to process resumes. Please ensure the backend server is running on port 5000.');
    } finally {
      setProcessing(false);
    }
  };

  const filteredCandidates = candidates.filter(c => {
    if (c.score < filters.minScore) return false;
    if (filters.skills && !c.skills.some(s => s.toLowerCase().includes(filters.skills.toLowerCase()))) return false;
    if (filters.location && !c.location.toLowerCase().includes(filters.location.toLowerCase())) return false;
    return true;
  });

  const getScoreColor = (score) => {
    if (score >= 85) return 'text-green-600 bg-green-100';
    if (score >= 70) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const exportToJSON = () => {
    const exportData = {
      job_title: jobTitle,
      export_date: new Date().toISOString(),
      total_candidates: candidates.length,
      candidates: candidates
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `screening_results_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const UploadSection = () => (
    <div className="space-y-6">
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-red-800 font-medium">Error</p>
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Briefcase className="w-5 h-5" />
          Job Description
        </h2>
        <textarea
          className="w-full h-64 p-4 border rounded-lg font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="Paste job description here..."
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        />
        <button
          onClick={() => setJobDescription(demoJobDescription)}
          className="mt-2 px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors"
        >
          Load Demo Job Description
        </button>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Upload className="w-5 h-5" />
          Upload Resumes
        </h2>
        
        <div
          className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center transition-colors hover:border-blue-400"
          onDrop={handleFileDrop}
          onDragOver={handleDragOver}
        >
          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p className="text-gray-600 mb-2">Drag and drop resumes here</p>
          <p className="text-sm text-gray-500 mb-4">Supports PDF and DOCX (max 10MB each)</p>
          <label className="inline-block">
            <input
              type="file"
              multiple
              accept=".pdf,.docx,.doc"
              onChange={handleFileSelect}
              className="hidden"
            />
            <span className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer inline-block transition-colors">
              Select Files
            </span>
          </label>
        </div>

        {selectedFiles.length > 0 && (
          <div className="mt-4">
            <h3 className="font-medium mb-2">Selected Files ({selectedFiles.length}):</h3>
            <ul className="space-y-1">
              {selectedFiles.map((file, idx) => (
                <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-600" />
                  {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="mt-6">
          <button
            onClick={handleProcessResumes}
            disabled={processing || !jobDescription || selectedFiles.length === 0}
            className="w-full px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
          >
            {processing ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                Processing Resumes... (This may take a minute)
              </>
            ) : (
              <>
                <TrendingUp className="w-5 h-5" />
                Process & Rank Candidates
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );

  const DashboardSection = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex flex-wrap gap-4 items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-semibold flex items-center gap-2">
              <User className="w-5 h-5" />
              Ranked Candidates for {jobTitle}
            </h2>
            <p className="text-sm text-gray-500 mt-1">
              {filteredCandidates.length} candidates {filters.minScore > 0 || filters.skills || filters.location ? '(filtered)' : ''}
            </p>
          </div>
          <button 
            onClick={exportToJSON}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 transition-colors"
          >
            <Download className="w-4 h-4" />
            Export Report
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-500" />
            <input
              type="number"
              placeholder="Min Score (0-100)"
              className="flex-1 px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              value={filters.minScore}
              onChange={(e) => setFilters({...filters, minScore: Number(e.target.value)})}
            />
          </div>
          <input
            type="text"
            placeholder="Filter by skill"
            className="px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={filters.skills}
            onChange={(e) => setFilters({...filters, skills: e.target.value})}
          />
          <input
            type="text"
            placeholder="Filter by location"
            className="px-3 py-2 border rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            value={filters.location}
            onChange={(e) => setFilters({...filters, location: e.target.value})}
          />
        </div>

        {filteredCandidates.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <User className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No candidates match your filters</p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredCandidates.map((candidate, idx) => (
              <div
                key={candidate.id}
                className="border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                onClick={() => setSelectedCandidate(candidate)}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="text-gray-500 font-medium">#{idx + 1}</span>
                      <h3 className="text-lg font-semibold">{candidate.name}</h3>
                      <span className={`px-3 py-1 rounded-full text-sm font-semibold ${getScoreColor(candidate.score)}`}>
                        {candidate.score}% Match
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{candidate.summary || 'No summary available'}</p>
                    <div className="flex flex-wrap gap-4 text-sm text-gray-500">
                      <span className="flex items-center gap-1">
                        <Briefcase className="w-4 h-4" />
                        {candidate.experience_years} years
                      </span>
                      {candidate.location && (
                        <span className="flex items-center gap-1">
                          <MapPin className="w-4 h-4" />
                          {candidate.location}
                        </span>
                      )}
                      <span className="flex items-center gap-1">
                        <Award className="w-4 h-4" />
                        {candidate.certifications.length} certs
                      </span>
                    </div>
                  </div>
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {candidate.matched_skills.slice(0, 6).map(skill => (
                    <span key={skill} className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded">
                      {skill}
                    </span>
                  ))}
                  {candidate.matched_skills.length > 6 && (
                    <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
                      +{candidate.matched_skills.length - 6} more
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const CandidateDetailModal = () => {
    if (!selectedCandidate) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={() => setSelectedCandidate(null)}>
        <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
          <div className="p-6">
            <div className="flex justify-between items-start mb-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">{selectedCandidate.name}</h2>
                <div className="flex flex-wrap gap-3 text-sm text-gray-600">
                  {selectedCandidate.email && <span>{selectedCandidate.email}</span>}
                  {selectedCandidate.phone && <span>{selectedCandidate.phone}</span>}
                  {selectedCandidate.location && (
                    <span className="flex items-center gap-1">
                      <MapPin className="w-4 h-4" />
                      {selectedCandidate.location}
                    </span>
                  )}
                </div>
              </div>
              <div className={`px-4 py-2 rounded-lg text-xl font-bold ${getScoreColor(selectedCandidate.score)}`}>
                {selectedCandidate.score}%
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="bg-green-50 rounded-lg p-4">
                <h3 className="font-semibold mb-3 flex items-center gap-2 text-green-800">
                  <CheckCircle className="w-5 h-5" />
                  Matched Skills ({selectedCandidate.matched_skills.length})
                </h3>
                <div className="flex flex-wrap gap-2">
                  {selectedCandidate.matched_skills.length > 0 ? (
                    selectedCandidate.matched_skills.map(skill => (
                      <span key={skill} className="px-3 py-1 bg-green-200 text-green-800 text-sm rounded">
                        {skill}
                      </span>
                    ))
                  ) : (
                    <p className="text-sm text-gray-600">No matched skills found</p>
                  )}
                </div>
              </div>

              <div className="bg-red-50 rounded-lg p-4">
                <h3 className="font-semibold mb-3 flex items-center gap-2 text-red-800">
                  <XCircle className="w-5 h-5" />
                  Missing Skills ({selectedCandidate.missing_skills.length})
                </h3>
                <div className="flex flex-wrap gap-2">
                  {selectedCandidate.missing_skills.length > 0 ? (
                    selectedCandidate.missing_skills.map(skill => (
                      <span key={skill} className="px-3 py-1 bg-red-200 text-red-800 text-sm rounded">
                        {skill}
                      </span>
                    ))
                  ) : (
                    <p className="text-sm text-green-600">All required skills present!</p>
                  )}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-blue-50 rounded-lg p-4">
                <h3 className="font-semibold mb-2 flex items-center gap-2 text-blue-800">
                  <Briefcase className="w-5 h-5" />
                  Experience Analysis
                </h3>
                <p className="text-sm text-gray-700">{selectedCandidate.experience_match}</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-semibold mb-2">Education</h3>
                <p className="text-sm text-gray-700">{selectedCandidate.education}</p>
              </div>

              {selectedCandidate.certifications.length > 0 && (
                <div className="bg-purple-50 rounded-lg p-4">
                  <h3 className="font-semibold mb-2 flex items-center gap-2 text-purple-800">
                    <Award className="w-5 h-5" />
                    Certifications ({selectedCandidate.certifications.length})
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedCandidate.certifications.map(cert => (
                      <span key={cert} className="px-3 py-1 bg-purple-200 text-purple-800 text-sm rounded">
                        {cert}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {selectedCandidate.bias_flags && selectedCandidate.bias_flags.length > 0 && (
                <div className="bg-yellow-50 rounded-lg p-4">
                  <h3 className="font-semibold mb-2 flex items-center gap-2 text-yellow-800">
                    <AlertCircle className="w-5 h-5" />
                    Bias Detection Notes
                  </h3>
                  <ul className="text-sm text-gray-700 list-disc list-inside">
                    {selectedCandidate.bias_flags.map((flag, i) => (
                      <li key={i}>{flag}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="mt-6 flex gap-3">
              <button className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
                Move to Interview
              </button>
              <button className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors">
                Request More Info
              </button>
              <button
                onClick={() => setSelectedCandidate(null)}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-gray-900">AI Resume Screening System</h1>
          <p className="text-sm text-gray-600">Intelligent candidate matching with bias-aware ranking</p>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="mb-6 flex gap-4">
          <button
            onClick={() => setActiveTab('upload')}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'upload'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
          >
            Upload & Process
          </button>
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`px-6 py-2 rounded-lg font-medium transition-colors ${
              activeTab === 'dashboard'
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-50'
            }`}
            disabled={candidates.length === 0}
          >
            Candidate Dashboard ({candidates.length})
          </button>
        </div>

        {activeTab === 'upload' && <UploadSection />}
        {activeTab === 'dashboard' && <DashboardSection />}
      </div>

      <CandidateDetailModal />

      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-gray-600">
            <div>
              <h3 className="font-semibold mb-2">Backend Status</h3>
              <ul className="space-y-1">
                <li>• API Endpoint: {API_URL}/api/screen</li>
                <li>• Semantic Model: SBERT (MiniLM)</li>
                <li>• CORS: Enabled for localhost:5173</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Features</h3>
              <ul className="space-y-1">
                <li>• Multi-format resume parsing</li>
                <li>• Semantic skill matching</li>
                <li>• Bias-aware ranking</li>
                <li>• Explainable AI decisions</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Requirements</h3>
              <ul className="space-y-1">
                <li>• Flask backend on port 5000</li>
                <li>• Python 3.8+ with dependencies</li>
                <li>• PDF/DOCX file support</li>
                <li>• ~2GB RAM for AI models</li>
              </ul>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default ResumeScreeningSystem;
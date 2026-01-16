"""
Flask API Backend for Resume Screening System
Handles file uploads, processing, and returns JSON-serializable results
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import json
import re
import logging

# Document parsing
import PyPDF2
import docx

# NLP and ML
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# CORS Configuration - Allow requests from React dev server
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# File upload configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


# ============================================================================
# DATA MODELS (Simplified for JSON serialization)
# ============================================================================

@dataclass
class PersonalInfo:
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""


@dataclass
class Education:
    degree: str
    institution: str
    year: Optional[str] = None


@dataclass
class WorkExperience:
    title: str
    company: str
    duration: str
    description: str


@dataclass
class CandidateProfile:
    personal_info: PersonalInfo
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)
    work_experience: List[WorkExperience] = field(default_factory=list)
    summary: str = ""
    total_experience_years: float = 0.0
    raw_text: str = ""


@dataclass
class JobRequirements:
    title: str
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    required_education: str = ""
    description: str = ""


# ============================================================================
# RESUME PARSER
# ============================================================================

class ResumeParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        self.all_skills = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express',
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp',
            'git', 'jira', 'agile', 'scrum', 'ci/cd', 'restful api', 'graphql'
        }
    
    def parse_pdf(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            return ""
    
    def parse_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            return ""
    
    def parse_file(self, file_path: str) -> str:
        path = Path(file_path)
        if path.suffix.lower() == '.pdf':
            return self.parse_pdf(file_path)
        elif path.suffix.lower() in ['.docx', '.doc']:
            return self.parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def extract_email(self, text: str) -> str:
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else ""
    
    def extract_phone(self, text: str) -> str:
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        matches = re.findall(phone_pattern, text)
        return matches[0] if matches else ""
    
    def extract_name(self, text: str) -> str:
        doc = self.nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line.split()) <= 4 and not any(char.isdigit() for char in line):
                return line
        return "Unknown"
    
    def extract_skills(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_skills = set()
        
        for skill in self.all_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill.title())
        
        return sorted(list(found_skills))
    
    def extract_certifications(self, text: str) -> List[str]:
        cert_pattern = r'\b(?:AWS|Azure|Google Cloud|PMP|CISSP|CKA|Scrum Master|Six Sigma)[^\n]*'
        matches = re.finditer(cert_pattern, text, re.IGNORECASE)
        certifications = set()
        for match in matches:
            cert_text = match.group(0).strip()
            if cert_text and len(cert_text) < 100:
                certifications.add(cert_text)
        return sorted(list(certifications))
    
    def extract_education(self, text: str) -> List[Education]:
        education_list = []
        degree_patterns = [
            r'(?:Bachelor|B\.S\.|B\.A\.|Master|M\.S\.|M\.A\.|MBA|Ph\.D\.|PhD)[^\n]*'
        ]
        
        for pattern in degree_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                degree_text = match.group(0).strip()
                institution = "Unknown"
                year = None
                
                context_start = max(0, match.start() - 100)
                context_end = min(len(text), match.end() + 100)
                context = text[context_start:context_end]
                
                uni_pattern = r'(?:University|College|Institute) of [A-Za-z\s]+'
                uni_match = re.search(uni_pattern, context, re.IGNORECASE)
                if uni_match:
                    institution = uni_match.group(0)
                
                year_pattern = r'\b(19|20)\d{2}\b'
                year_match = re.search(year_pattern, context)
                if year_match:
                    year = year_match.group(0)
                
                education_list.append(Education(
                    degree=degree_text,
                    institution=institution,
                    year=year
                ))
        
        return education_list
    
    def extract_work_experience(self, text: str) -> Tuple[List[WorkExperience], float]:
        experiences = []
        total_years = 0.0
        
        experience_pattern = r'(?:^|\n)([A-Z][A-Za-z\s&,]+(?:Engineer|Developer|Manager|Analyst))[,\s]*\n?([A-Z][A-Za-z\s&,\.]+)[,\s]*\n?((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[0-9]{4})[^\n]*)'
        
        matches = re.finditer(experience_pattern, text, re.MULTILINE)
        
        for match in matches:
            title = match.group(1).strip()
            company = match.group(2).strip()
            duration = match.group(3).strip()
            
            desc_start = match.end()
            desc_end = min(desc_start + 300, len(text))
            description = text[desc_start:desc_end].split('\n\n')[0].strip()
            
            experiences.append(WorkExperience(
                title=title,
                company=company,
                duration=duration,
                description=description
            ))
            
            years_match = re.search(r'(\d+)\s*(?:years?|yrs?)', duration, re.IGNORECASE)
            if years_match:
                total_years += float(years_match.group(1))
        
        if total_years == 0 and experiences:
            total_years = len(experiences) * 2.0
        
        return experiences, total_years
    
    def parse_resume(self, file_path: str) -> CandidateProfile:
        logger.info(f"Parsing resume: {file_path}")
        text = self.parse_file(file_path)
        
        if not text:
            raise ValueError(f"Failed to extract text from {file_path}")
        
        name = self.extract_name(text)
        email = self.extract_email(text)
        phone = self.extract_phone(text)
        skills = self.extract_skills(text)
        certifications = self.extract_certifications(text)
        education = self.extract_education(text)
        work_experience, total_years = self.extract_work_experience(text)
        
        location = ""
        location_pattern = r'([A-Z][a-z]+,\s*[A-Z]{2})'
        location_match = re.search(location_pattern, text)
        if location_match:
            location = location_match.group(1)
        
        summary_lines = [line.strip() for line in text.split('\n') if line.strip()]
        summary = ' '.join(summary_lines[1:3]) if len(summary_lines) > 1 else ""
        summary = summary[:300]
        
        personal_info = PersonalInfo(
            name=name,
            email=email,
            phone=phone,
            location=location
        )
        
        profile = CandidateProfile(
            personal_info=personal_info,
            skills=skills,
            certifications=certifications,
            education=education,
            work_experience=work_experience,
            summary=summary,
            total_experience_years=total_years,
            raw_text=text
        )
        
        return profile


# ============================================================================
# JOB PARSER
# ============================================================================

class JobDescriptionParser:
    def parse_job_description(self, text: str) -> JobRequirements:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        title = lines[0] if lines else "Unknown Position"
        
        required_skills = self._extract_required_skills(text)
        preferred_skills = self._extract_preferred_skills(text)
        min_experience = self._extract_experience_requirement(text)
        required_education = self._extract_education_requirement(text)
        
        return JobRequirements(
            title=title,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            min_experience_years=min_experience,
            required_education=required_education,
            description=text
        )
    
    def _extract_required_skills(self, text: str) -> List[str]:
        skills = set()
        required_pattern = r'(?:required|must have|essential)[:\s]+((?:.|\n)*?)(?:\n\n|preferred|nice to have|\Z)'
        required_match = re.search(required_pattern, text, re.IGNORECASE)
        
        if required_match:
            skills_text = required_match.group(1)
            potential_skills = re.split(r'[,•\n-]', skills_text)
            for skill in potential_skills:
                skill = skill.strip()
                if skill and 3 <= len(skill) <= 50:
                    skills.add(skill)
        
        return sorted(list(skills))
    
    def _extract_preferred_skills(self, text: str) -> List[str]:
        skills = set()
        preferred_pattern = r'(?:preferred|nice to have|bonus)[:\s]+((?:.|\n)*?)(?:\n\n|education|\Z)'
        preferred_match = re.search(preferred_pattern, text, re.IGNORECASE)
        
        if preferred_match:
            skills_text = preferred_match.group(1)
            potential_skills = re.split(r'[,•\n-]', skills_text)
            for skill in potential_skills:
                skill = skill.strip()
                if skill and 3 <= len(skill) <= 50:
                    skills.add(skill)
        
        return sorted(list(skills))
    
    def _extract_experience_requirement(self, text: str) -> float:
        experience_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?) of experience',
            r'experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        for pattern in experience_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 0.0
    
    def _extract_education_requirement(self, text: str) -> str:
        degree_pattern = r'(?:Bachelor|Master|PhD)[^\n]*'
        degree_match = re.search(degree_pattern, text, re.IGNORECASE)
        if degree_match:
            return degree_match.group(0)
        return ""


# ============================================================================
# SEMANTIC MATCHER
# ============================================================================

class SemanticMatcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_cache = {}
    
    def get_embedding(self, text: str) -> np.ndarray:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        self.embedding_cache[text] = embedding
        return embedding
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        similarity = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def match_profile_to_job(self, profile: CandidateProfile, job: JobRequirements) -> float:
        candidate_text = f"{profile.summary} {' '.join(profile.skills)}"
        job_text = f"{job.description} {' '.join(job.required_skills)}"
        return self.compute_similarity(candidate_text, job_text)


# ============================================================================
# SCORING ENGINE
# ============================================================================

class ScoringEngine:
    def __init__(self, semantic_matcher: SemanticMatcher):
        self.semantic_matcher = semantic_matcher
        self.weights = {
            'skill_match': 0.35,
            'experience': 0.25,
            'semantic_similarity': 0.20,
            'education': 0.10,
            'certifications': 0.10
        }
    
    def compute_skill_score(self, candidate_skills: List[str], required_skills: List[str], 
                           preferred_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        candidate_skills_lower = set([s.lower() for s in candidate_skills])
        required_skills_lower = set([s.lower() for s in required_skills])
        preferred_skills_lower = set([s.lower() for s in preferred_skills])
        
        matched_required = candidate_skills_lower & required_skills_lower
        missing_required = required_skills_lower - candidate_skills_lower
        matched_preferred = candidate_skills_lower & preferred_skills_lower
        
        if not required_skills_lower:
            required_score = 1.0
        else:
            required_score = len(matched_required) / len(required_skills_lower)
        
        if preferred_skills_lower:
            preferred_score = len(matched_preferred) / len(preferred_skills_lower)
            skill_score = 0.8 * required_score + 0.2 * preferred_score
        else:
            skill_score = required_score
        
        matched_skills = [s for s in candidate_skills if s.lower() in matched_required | matched_preferred]
        missing_skills = [s for s in required_skills if s.lower() in missing_required]
        
        return skill_score, matched_skills, missing_skills
    
    def compute_experience_score(self, candidate_years: float, required_years: float) -> Tuple[float, str]:
        if required_years == 0:
            return 1.0, "No specific requirement"
        
        ratio = candidate_years / required_years
        
        if ratio >= 1.5:
            score = 1.0
            explanation = f"Excellent - {candidate_years:.0f} years exceeds requirement"
        elif ratio >= 1.0:
            score = 1.0
            explanation = f"Good - {candidate_years:.0f} years meets {required_years:.0f} year requirement"
        elif ratio >= 0.8:
            score = 0.85
            explanation = f"Acceptable - {candidate_years:.0f} years slightly below requirement"
        else:
            score = 0.65
            explanation = f"Below requirement - {candidate_years:.0f} years vs {required_years:.0f} required"
        
        return score, explanation
    
    def score_candidate(self, profile: CandidateProfile, job: JobRequirements, 
                       candidate_id: str) -> Dict[str, Any]:
        logger.info(f"Scoring candidate: {profile.personal_info.name}")
        
        skill_score, matched_skills, missing_skills = self.compute_skill_score(
            profile.skills, job.required_skills, job.preferred_skills
        )
        
        experience_score, experience_explanation = self.compute_experience_score(
            profile.total_experience_years, job.min_experience_years
        )
        
        education_score = 0.8 if profile.education else 0.5
        cert_score = 0.8 if profile.certifications else 0.5
        
        semantic_similarity = self.semantic_matcher.match_profile_to_job(profile, job)
        
        overall_score = (
            self.weights['skill_match'] * skill_score +
            self.weights['experience'] * experience_score +
            self.weights['semantic_similarity'] * semantic_similarity +
            self.weights['education'] * education_score +
            self.weights['certifications'] * cert_score
        )
        
        # Return simple JSON-serializable dict
        return {
            'id': candidate_id,
            'name': profile.personal_info.name,
            'email': profile.personal_info.email,
            'phone': profile.personal_info.phone,
            'location': profile.personal_info.location,
            'score': round(overall_score * 100, 1),
            'experience_years': profile.total_experience_years,
            'skills': profile.skills,
            'education': profile.education[0].degree if profile.education else "Not specified",
            'certifications': profile.certifications,
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'experience_match': experience_explanation,
            'bias_flags': [],
            'summary': profile.summary
        }


# ============================================================================
# GLOBAL PIPELINE INSTANCE (Initialized once for performance)
# ============================================================================

resume_parser = None
job_parser = None
semantic_matcher = None
scoring_engine = None

def initialize_pipeline():
    global resume_parser, job_parser, semantic_matcher, scoring_engine
    if resume_parser is None:
        logger.info("Initializing pipeline components...")
        resume_parser = ResumeParser()
        job_parser = JobDescriptionParser()
        semantic_matcher = SemanticMatcher()
        scoring_engine = ScoringEngine(semantic_matcher)
        logger.info("Pipeline initialized successfully")


# ============================================================================
# API ENDPOINTS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Resume screening API is running'}), 200


@app.route('/api/screen', methods=['POST', 'OPTIONS'])
def screen_resumes():
    """Main endpoint for resume screening"""
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        # Initialize pipeline if needed
        initialize_pipeline()
        
        # Validate request
        if 'jobDescription' not in request.form:
            return jsonify({'error': 'Missing jobDescription field'}), 400
        
        if 'resumes' not in request.files:
            return jsonify({'error': 'No resume files uploaded'}), 400
        
        job_description = request.form['jobDescription']
        resume_files = request.files.getlist('resumes')
        
        if not resume_files:
            return jsonify({'error': 'No resume files provided'}), 400
        
        logger.info(f"Processing {len(resume_files)} resumes")
        
        # Parse job description
        job_requirements = job_parser.parse_job_description(job_description)
        
        # Process each resume
        results = []
        temp_files = []
        
        for idx, file in enumerate(resume_files):
            if file and allowed_file(file.filename):
                # Save file temporarily
                filename = secure_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{idx}_{filename}")
                file.save(temp_path)
                temp_files.append(temp_path)
                
                try:
                    # Parse resume
                    profile = resume_parser.parse_resume(temp_path)
                    
                    # Score candidate
                    candidate_id = f"candidate_{idx + 1}"
                    score_result = scoring_engine.score_candidate(profile, job_requirements, candidate_id)
                    
                    results.append(score_result)
                    
                except Exception as e:
                    logger.error(f"Error processing resume {filename}: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Continue processing other resumes
                    results.append({
                        'id': f"candidate_{idx + 1}",
                        'name': f"Error - {filename}",
                        'email': '',
                        'phone': '',
                        'location': '',
                        'score': 0,
                        'experience_years': 0,
                        'skills': [],
                        'education': 'Processing failed',
                        'certifications': [],
                        'matched_skills': [],
                        'missing_skills': [],
                        'experience_match': f'Error: {str(e)}',
                        'bias_flags': ['Processing error'],
                        'summary': ''
                    })
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Successfully processed {len(results)} resumes")
        
        return jsonify({
            'success': True,
            'job_title': job_requirements.title,
            'total_candidates': len(results),
            'candidates': results
        }), 200
        
    except Exception as e:
        logger.error(f"Screening error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify API is working"""
    return jsonify({
        'message': 'API is working',
        'endpoints': {
            'health': '/api/health',
            'screen': '/api/screen (POST)',
            'test': '/api/test'
        }
    }), 200


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting Resume Screening API server...")
    logger.info("Server will run on http://localhost:5000")
    logger.info("CORS enabled for http://localhost:5173")
    app.run(debug=True, host='0.0.0.0', port=5000)
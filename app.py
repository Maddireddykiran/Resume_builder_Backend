from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import openai
import json
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes with additional options
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"]}})

# Set OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables. API calls will fail.")
else:
    openai.api_key = openai_api_key

@app.route('/', methods=['GET'])
def home():
    """Basic health check endpoint"""
    return jsonify({
        "status": "online",
        "message": "Resume analysis API is running",
        "api_key_configured": bool(openai_api_key)
    })

@app.route('/api/analyze', methods=['HEAD', 'OPTIONS'])
def check_status():
    # This endpoint is used to check if the backend is running
    return '', 200

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Validate API key is available
        if not openai_api_key:
            return jsonify({
                'error': 'OpenAI API key not found. Please check your .env file.',
                'details': 'Add OPENAI_API_KEY=your_key_here to your .env file'
            }), 500
            
        data = request.json
        
        if not data or 'resume' not in data or 'jobDescription' not in data:
            return jsonify({'error': 'Missing resume or job description data'}), 400
        
        resume_data = data['resume']
        job_description = data['jobDescription']
        
        # Convert resume data to a readable format for the AI
        resume_text = format_resume_for_ai(resume_data)
        
        try:
            # Call OpenAI API to analyze and generate tailored content
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a professional resume consultant specializing in tailoring resumes to specific job descriptions. Your task is to analyze the candidate's resume and the job description to create highly tailored content that maximizes the match between the candidate's qualifications and the job requirements.

Follow these guidelines:
1. Identify keywords and skills in the job description
2. Match these with relevant experiences in the resume
3. Use industry-specific terminology from the job description
4. Quantify achievements whenever possible
5. Focus on achievements and skills that directly relate to the job requirements
6. Maintain professionalism and accuracy at all times
7. Don't fabricate experience, but optimize the wording of existing experience
8. NEVER inflate years of experience - use ONLY the timeframes and durations that appear in the original resume
9. Do NOT claim expertise or experience levels that aren't supported by the resume
10. If the job requires more experience than the candidate has, focus on relevant achievements instead of claiming that experience"""},
                    {"role": "user", "content": f"""
Below is a resume and a job description. Create a tailored version of the resume that highlights relevant skills and experiences matching this specific job.

---RESUME---
{resume_text}

---JOB DESCRIPTION---
{job_description}

Please provide:

1. PROFESSIONAL SUMMARY (required):
   - Create a concise, powerful summary (3-4 sentences) that positions the candidate perfectly for this job
   - Highlight the most relevant skills, experience, and achievements that match the job requirements
   - Use industry-specific terminology from the job description
   - Focus on quantifiable achievements and ACTUAL years of relevant experience (do NOT inflate experience)
   - Ensure 100% accuracy - no fabrication, only optimization of existing information

2. WORK EXPERIENCE (required):
   - For each company in the original resume, create 3-4 bullet points that are specifically tailored to this job
   - Use strong action verbs at the beginning of each bullet point
   - Include specific keywords from the job description
   - Quantify achievements with metrics whenever possible (%, $, numbers)
   - Focus on accomplishments rather than responsibilities
   - Address specific requirements mentioned in the job description
   - Make sure each bullet point is relevant to the job being applied for
   - IMPORTANT: Do NOT invent new experience or inflate existing experience - stay strictly within what is presented in the original resume
   - If the job requires skills not explicitly stated in the resume, focus on transferable skills instead of claiming direct experience

Format your response exactly as follows:

SUMMARY:
[Your tailored summary here - no bullet points, just a paragraph]

EXPERIENCE:
[Company Name #1]
• [Tailored bullet point 1]
• [Tailored bullet point 2]
• [Tailored bullet point 3]
• [Tailored bullet point 4 if applicable]

[Company Name #2]
• [Tailored bullet point 1]
• [Tailored bullet point 2]
• [Tailored bullet point 3]
• [Tailored bullet point 4 if applicable]

(continue for all work experiences in the original resume)
"""}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            ai_content = response.choices[0].message['content']
            
            # Parse AI response to extract summary and work experience
            tailored_content = parse_ai_response(ai_content)
            
            return jsonify({
                'success': True,
                'tailoredContent': tailored_content
            })
        except openai.error.OpenAIError as oe:
            print(f"OpenAI API Error: {str(oe)}")
            return jsonify({
                'error': f'OpenAI API error: {str(oe)}',
                'success': False,
                'tailoredContent': None
            }), 500
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False,
            'tailoredContent': None
        }), 500

def format_resume_for_ai(resume_data):
    """Format resume data into a readable text for the AI"""
    formatted = []
    
    # Add profile information
    if 'profile' in resume_data:
        profile = resume_data['profile']
        formatted.append("PROFILE:")
        formatted.append(f"Name: {profile.get('name', 'N/A')}")
        formatted.append(f"Email: {profile.get('email', 'N/A')}")
        formatted.append(f"Phone: {profile.get('phone', 'N/A')}")
        formatted.append(f"Location: {profile.get('location', 'N/A')}")
        formatted.append(f"URL: {profile.get('url', 'N/A')}")
        formatted.append(f"Summary: {profile.get('summary', 'N/A')}")
        formatted.append("")
    
    # Add work experience
    if 'workExperiences' in resume_data and resume_data['workExperiences']:
        formatted.append("WORK EXPERIENCE:")
        for exp in resume_data['workExperiences']:
            formatted.append(f"Company: {exp.get('company', 'N/A')}")
            formatted.append(f"Job Title: {exp.get('jobTitle', 'N/A')}")
            formatted.append(f"Date: {exp.get('date', 'N/A')}")
            if 'descriptions' in exp and exp['descriptions']:
                formatted.append("Descriptions:")
                for desc in exp['descriptions']:
                    formatted.append(f"- {desc}")
            formatted.append("")
    
    # Add education
    if 'educations' in resume_data and resume_data['educations']:
        formatted.append("EDUCATION:")
        for edu in resume_data['educations']:
            formatted.append(f"School: {edu.get('school', 'N/A')}")
            formatted.append(f"Degree: {edu.get('degree', 'N/A')}")
            formatted.append(f"Date: {edu.get('date', 'N/A')}")
            if 'descriptions' in edu and edu['descriptions']:
                formatted.append("Descriptions:")
                for desc in edu['descriptions']:
                    formatted.append(f"- {desc}")
            formatted.append("")
    
    # Add skills
    if 'skills' in resume_data:
        formatted.append("SKILLS:")
        if 'featuredSkills' in resume_data['skills']:
            for skill in resume_data['skills']['featuredSkills']:
                if skill.get('skill'):
                    formatted.append(f"- {skill.get('skill')}")
        
        if 'descriptions' in resume_data['skills']:
            for desc in resume_data['skills']['descriptions']:
                formatted.append(f"- {desc}")
        formatted.append("")
    
    # Add projects
    if 'projects' in resume_data and resume_data['projects']:
        formatted.append("PROJECTS:")
        for proj in resume_data['projects']:
            formatted.append(f"Project: {proj.get('project', 'N/A')}")
            formatted.append(f"Date: {proj.get('date', 'N/A')}")
            if 'descriptions' in proj and proj['descriptions']:
                formatted.append("Descriptions:")
                for desc in proj['descriptions']:
                    formatted.append(f"- {desc}")
            formatted.append("")
    
    return "\n".join(formatted)

def parse_ai_response(ai_response):
    """Parse the AI response to extract summary and work experience sections"""
    # Initialize with default structure
    result = {
        'summary': '',
        'workExperience': []
    }
    
    # Extract summary section (everything between "SUMMARY:" and "EXPERIENCE:")
    summary_match = re.search(r'SUMMARY:(.*?)(?=EXPERIENCE:|$)', ai_response, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary_text = summary_match.group(1).strip()
        # Clean up the summary text
        summary_text = re.sub(r'^\s*•\s*', '', summary_text)  # Remove bullet points if present
        summary_text = re.sub(r'\s+', ' ', summary_text)  # Replace multiple spaces with single space
        result['summary'] = summary_text
    
    # Extract experience section
    experience_section = re.search(r'EXPERIENCE:(.*?)$', ai_response, re.DOTALL | re.IGNORECASE)
    if experience_section:
        experience_text = experience_section.group(1).strip()
        
        # Split by company (companies are lines that don't start with a bullet point)
        company_sections = re.split(r'\n(?=[^\s•\-*])', experience_text)
        
        for section in company_sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            company_name = lines[0].strip()
            bullet_points = []
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    # Extract the bullet point text
                    point = line[1:].strip()
                    if point:
                        bullet_points.append(point)
            
            if company_name and bullet_points:
                result['workExperience'].append({
                    'company': company_name,
                    'bulletPoints': bullet_points[:4]  # Limit to 4 bullet points
                })
    
    # Remove any special characters from summary and bullet points
    result['summary'] = re.sub(r'[*"]', '', result['summary'])
    for exp in result['workExperience']:
        exp['bulletPoints'] = [re.sub(r'[*"]', '', bp) for bp in exp['bulletPoints']]
    
    return result

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    print(f"OpenAI API key configured: {bool(openai_api_key)}")
    app.run(host='0.0.0.0', port=port)
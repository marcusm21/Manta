#!/usr/bin/env python3
# Manta Insights - AI-Powered Hackathon Analysis System 

import pandas as pd
import numpy as np
import requests
from github import Github
import os
import json
from dotenv import load_dotenv
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
import time
from datetime import datetime

load_dotenv()

class MantaInsights:
    """Manta Insights: AI-Powered Hackathon Analysis System"""
    
    VERSION = "1.0.1"
    
    def __init__(self, github_token, anthropic_api_key, verbose=False):
        """Initialize Manta Insights with API access"""
        self.g = Github(github_token)
        self.verbose = verbose
        
        # Set up logging
        level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MantaInsights")
        
        # Initialize LLM client
        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required for Manta Insights")
            
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        self.logger.info(f"Manta Insights v{self.VERSION} initialized")
        
    def analyze_submissions(self, input_file, output_dir="manta_output", max_repos=None):
        """Main analysis method for processing hackathon submissions"""
        start_time = datetime.now()
        
        self.logger.info(f"Starting analysis of {input_file}")
        self.logger.info(f"Output will be saved to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process submissions
        df = self._load_submission_data(input_file)
        
        # Limit repositories if specified (for testing)
        if max_repos and max_repos > 0:
            self.logger.info(f"Limiting analysis to {max_repos} repositories")
            df = df.head(max_repos)
            
        df = self._clean_submission_data(df)
        
        # Process GitHub repositories
        if 'github_url' in df.columns:
            df = self._process_github_data(df)
        else:
            self.logger.warning("No 'github_url' column found in input file")
        
        # Run LLM analysis
        self.logger.info("Running LLM analysis on submissions...")
        df = self._analyze_with_llm(df)
        
        # Save results
        output_path = os.path.join(output_dir, "manta_analysis.csv")
        df.to_csv(output_path, index=False)
        
        # Generate reports
        self._generate_reports(df, output_dir)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.logger.info(f"Analysis completed in {duration:.1f} seconds")
        self.logger.info(f"Processed {len(df)} submissions")
            
        return df
    
    def _load_submission_data(self, file_path):
        """Load submission data from CSV or Excel file"""
        self.logger.info(f"Loading submission data from {file_path}")
            
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading submission data: {str(e)}")
            raise RuntimeError(f"Error loading submission data: {str(e)}")
    
    def _clean_submission_data(self, df):
        """Clean and standardize the submission data"""
        self.logger.info("Cleaning and standardizing submission data")
            
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Handle missing values
        required_cols = ['team_name', 'project_name', 'description']
        for col in required_cols:
            if col in df.columns and df[col].isnull().any():
                self.logger.info(f"Filling missing values in {col} column")
                df[col] = df[col].fillna('Not provided')
        
        # Extract GitHub URLs if embedded in other fields
        if 'github_url' not in df.columns and 'project_url' in df.columns:
            self.logger.info("Extracting GitHub URLs from project_url column")
            df['github_url'] = df['project_url'].apply(
                lambda x: x if isinstance(x, str) and 'github.com' in x else None
            )
            
        return df
    
    def _process_github_data(self, df):
        """Process GitHub repositories to collect basic information"""
        self.logger.info(f"Processing {df['github_url'].notna().sum()} GitHub repositories")
        
        # Initialize columns
        df['repo_processed'] = False
        df['repo_error'] = None
        df['languages'] = df.apply(lambda x: [], axis=1)
        df['readme'] = None
        df['disqualified'] = False
        df['disqualification_reason'] = None
        
        # Process each repository
        for idx, row in df.iterrows():
            if pd.isna(row.get('github_url')) or not isinstance(row.get('github_url'), str):
                # Mark projects without GitHub URLs as disqualified
                df.at[idx, 'disqualified'] = True
                df.at[idx, 'disqualification_reason'] = "No valid GitHub URL provided"
                continue
                
            try:
                # Extract repo information
                repo_url = row['github_url']
                repo_parts = repo_url.rstrip('/').split('/')
                if len(repo_parts) < 5 or 'github.com' not in repo_parts:
                    df.at[idx, 'disqualified'] = True
                    df.at[idx, 'disqualification_reason'] = f"Invalid GitHub URL format: {repo_url}"
                    continue
                    
                repo_owner = repo_parts[-2]
                repo_name = repo_parts[-1]
                
                # Get repository
                repo = self.g.get_repo(f"{repo_owner}/{repo_name}")
                
                # Get languages
                df.at[idx, 'languages'] = list(repo.get_languages().keys())
                
                # Get README (truncated)
                try:
                    readme = repo.get_readme()
                    df.at[idx, 'readme'] = readme.decoded_content.decode('utf-8')[:5000]
                except:
                    self.logger.warning(f"Could not get README for {repo_owner}/{repo_name}")
                
                df.at[idx, 'repo_processed'] = True
                    
            except Exception as e:
                self.logger.error(f"Error processing repo at index {idx}: {str(e)}")
                df.at[idx, 'repo_error'] = str(e)
                df.at[idx, 'disqualified'] = True
                df.at[idx, 'disqualification_reason'] = f"Repository access error: {str(e)}"
        
        return df
    
    def _analyze_with_llm(self, df):
        """Analyze projects using LLM to provide insights"""
        # Initialize LLM analysis columns
        df['analysis'] = None
        df['project_type'] = None
        df['disqualified'] = df['disqualified'] if 'disqualified' in df.columns else False  
        df['disqualification_reason'] = df['disqualification_reason'] if 'disqualification_reason' in df.columns else None  
        df['innovation_score'] = 0.0
        df['technical_complexity'] = 0.0
        df['presentation_quality'] = 0.0
        df['overall_score'] = 0.0
        
        # Process each project
        for idx, row in df.iterrows():
            # Skip already disqualified projects
            if row.get('disqualified', False):
                self.logger.info(f"Skipping LLM analysis for disqualified project: {row.get('project_name', f'project_{idx}')}")
                continue
                
            try:
                self.logger.info(f"Analyzing project: {row.get('project_name', f'project_{idx}')}")
                
                # Create prompt with all available information
                prompt = self._create_analysis_prompt(row)
                
                # Generate analysis with the LLM
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                analysis = response.content[0].text
                
                # Extract scores, project type, and disqualification status from the analysis
                scores, project_type, disqualified, disqualification_reason = self._extract_scores(analysis)
                
                # Update DataFrame
                df.at[idx, 'analysis'] = analysis
                df.at[idx, 'project_type'] = project_type
                df.at[idx, 'disqualified'] = disqualified
                
                if disqualified and disqualification_reason:
                    df.at[idx, 'disqualification_reason'] = disqualification_reason
                    
                # Only set scores if not disqualified
                if not disqualified:
                    df.at[idx, 'innovation_score'] = scores.get('innovation', 0.0)
                    df.at[idx, 'technical_complexity'] = scores.get('technical', 0.0)
                    df.at[idx, 'presentation_quality'] = scores.get('presentation', 0.0)
                    df.at[idx, 'overall_score'] = scores.get('overall', 0.0)
                
                # Add delay to avoid rate limits
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error analyzing project at index {idx}: {str(e)}")
        
        return df
    
    def _create_analysis_prompt(self, project_row):
        """Create a comprehensive prompt for LLM to analyze the project"""
        
        project_name = project_row.get('project_name', 'Unnamed Project')
        team_name = project_row.get('team_name', 'Unknown Team')
        description = project_row.get('description', 'No description provided')
        
        # Get languages if available
        languages = project_row.get('languages', [])
        tech_text = ", ".join(languages) if languages else "Unknown"
        
        # Get README if available
        readme = project_row.get('readme', '')
        readme_text = f"README excerpt:\n{readme[:2000]}..." if readme else "No README available"
        
        # Create the prompt
        prompt = f"""
        You are evaluating a hackathon project. Please provide a comprehensive analysis and scoring.
        
        PROJECT DETAILS:
        Project Name: {project_name}
        Team: {team_name}
        Description: {description}
        Technologies: {tech_text}
        
        {readme_text}
        
        Please provide:

        1. CODE QUALITY CHECK: Examine the project description and README. If you detect clear evidence of any of the following issues, explain why and classify the project as 'DISQUALIFIED':
            - Plagiarism or stolen code
            - Submission of pre-existing projects without significant new development
            - Completely empty or non-functional repositories
            - Obviously fake/generated projects with no real implementation

        If none of these issues are detected, classify as 'QUALIFIED' and continue with the full analysis.
        
        2. PROJECT TYPE: Classify this project into exactly ONE of the following categories:
            - Gaming: Games, game platforms, gaming-related tools
            - DeFi: Decentralized finance, trading, financial applications
            - Infra: Infrastructure, developer tools, frameworks
            - Entertainment: Non-game entertainment, social, content creation
            - Security: Security tools, auditing, protection
            - Other: Projects that don't fit neatly in other categories

        3. ANALYSIS: A 300-word analysis of the project covering:
           - What the project does
           - Technical approach
           - Strengths and weaknesses
           - Innovative aspects
           - Potential improvements
        
        4. SCORES: Rate each category from 0-10
           - Innovation Score: How creative and novel is the project?
           - Technical Complexity: How technically challenging is the implementation?
           - Presentation Quality: How well is the project documented and presented?
           - Overall Score: Overall impression of the project
           
        5. EXECUTIVE SUMMARY: A 100-word summary highlighting the key points for judges
        
        FORMAT YOUR RESPONSE LIKE THIS:
        
        ## Qualification Status
        [QUALIFIED or DISQUALIFIED]
        [If disqualified, explain why]
        
        ## Project Type
        [Category name]
        
        ## Analysis
        [Your analysis here]
        
        ## Scores
        Innovation: X/10
        Technical Complexity: X/10
        Presentation Quality: X/10
        Overall: X/10
        
        ## Executive Summary
        [Your executive summary here]
        """
        
        return prompt
    
    def _extract_scores(self, analysis_text):
        """Extract numerical scores, project type, and qualification status from the analysis text"""
        scores = {
            'innovation': 0.0,
            'technical': 0.0,
            'presentation': 0.0,
            'overall': 0.0
        }
        
        project_type = "Other"  # Default value
        disqualified = False
        disqualification_reason = None
        
        try:
            # Look for qualification status
            qualification_section = False
            for line in analysis_text.split('\n'):
                line = line.strip()
                
                if line == "## Qualification Status":
                    qualification_section = True
                    continue
                    
                if qualification_section and line and not line.startswith("##"):
                    if "DISQUALIFIED" in line:
                        disqualified = True
                        disqualification_reason = line
                    break
            
            # If disqualified, get reason from next line(s)
            if disqualified and not disqualification_reason:
                capture_reason = False
                reason_lines = []
                for line in analysis_text.split('\n'):
                    line = line.strip()
                    if line == "DISQUALIFIED":
                        capture_reason = True
                        continue
                    if capture_reason and line and not line.startswith("##"):
                        reason_lines.append(line)
                    elif capture_reason and line.startswith("##"):
                        break
                if reason_lines:
                    disqualification_reason = " ".join(reason_lines)
            
            # Look for project type
            type_section = False
            for line in analysis_text.split('\n'):
                line = line.strip()
                
                if line == "## Project Type":
                    type_section = True
                    continue
                    
                if type_section and line and not line.startswith("##"):
                    project_type = line
                    break
                    
            # Look for scores in the format "Category: X/10"
            for line in analysis_text.split('\n'):
                line = line.strip()
                
                if line.startswith('Innovation:'):
                    score_text = line.split(':')[1].strip().split('/')[0]
                    scores['innovation'] = float(score_text)
                    
                elif line.startswith('Technical Complexity:'):
                    score_text = line.split(':')[1].strip().split('/')[0]
                    scores['technical'] = float(score_text)
                    
                elif line.startswith('Presentation Quality:'):
                    score_text = line.split(':')[1].strip().split('/')[0]
                    scores['presentation'] = float(score_text)
                    
                elif line.startswith('Overall:'):
                    score_text = line.split(':')[1].strip().split('/')[0]
                    scores['overall'] = float(score_text)
        except Exception as e:
            self.logger.warning(f"Could not extract scores/status from analysis text: {str(e)}")
                
        return scores, project_type, disqualified, disqualification_reason
    
    def _generate_reports(self, df, output_dir):
        """Generate project reports in Markdown format"""
        # Create reports directory
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create index file
        index_content = "# Manta Insights: Project Reports\n\n"
        index_content += f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Create separate sections for qualified and disqualified projects
        index_content += "## Qualified Projects\n\n"
        
        # Sort qualified projects by overall score
        qualified_df = df[~df['disqualified']].sort_values('overall_score', ascending=False)
        
        # Add qualified projects to index
        for idx, project in qualified_df.iterrows():
            project_name = project.get('project_name', f'project_{idx}')
            safe_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in project_name)
            score = project.get('overall_score', 0)
            index_content += f"- [{project_name}]({safe_name}.md) - Score: {score:.1f}/10\n"
        
        # Add disqualified projects section
        index_content += "\n## Disqualified Projects\n\n"
        
        # Get disqualified projects
        disqualified_df = df[df['disqualified']]
        
        # Add disqualified projects to index
        for idx, project in disqualified_df.iterrows():
            project_name = project.get('project_name', f'project_{idx}')
            safe_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in project_name)
            reason = project.get('disqualification_reason', 'Unknown reason')
            # Truncate reason if it's too long
            short_reason = reason[:100] + "..." if len(reason) > 100 else reason
            index_content += f"- [{project_name}]({safe_name}.md) - Reason: {short_reason}\n"
        
        # Create report for each project (both qualified and disqualified)
        for idx, project in df.iterrows():
            project_name = project.get('project_name', f'project_{idx}')
            safe_name = ''.join(c if c.isalnum() or c in ['-', '_'] else '_' for c in project_name)
            
            # Create report content
            report_content = f"# {project_name}\n\n"
            report_content += f"**Team:** {project.get('team_name', 'Unknown')}\n\n"
            
            # Add disqualification status
            if project.get('disqualified', False):
                report_content += f"**Status: DISQUALIFIED**\n\n"
                report_content += f"**Reason:** {project.get('disqualification_reason', 'No reason provided')}\n\n"
            else:
                report_content += f"**Status: QUALIFIED**\n\n"
                report_content += f"**Project Type:** {project.get('project_type', 'Not classified')}\n\n"
            
            # Add GitHub link if available
            if project.get('github_url'):
                report_content += f"**GitHub:** [{project.get('github_url')}]({project.get('github_url')})\n\n"
            
            # Add technologies if available and project is qualified
            if not project.get('disqualified', False) and project.get('languages') and len(project.get('languages')) > 0:
                report_content += f"**Technologies:** {', '.join(project.get('languages'))}\n\n"
            
            # Add analysis if available
            if project.get('analysis'):
                report_content += f"{project.get('analysis')}\n\n"
            
            # Add footer
            report_content += "---\n"
            report_content += f"Generated by Manta Insights v{self.VERSION} on {datetime.now().strftime('%Y-%m-%d')}\n"
            
            # Write report file
            with open(os.path.join(reports_dir, f"{safe_name}.md"), 'w', encoding='utf-8') as f:
                f.write(report_content)
        
        # Write index file
        with open(os.path.join(reports_dir, "index.md"), 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        self.logger.info(f"Generated {len(df)} project reports in {reports_dir}")


def main():
    """Command-line interface for Manta Insights"""
    parser = argparse.ArgumentParser(description="Manta Insights: AI-Powered Hackathon Analysis System")
    
    parser.add_argument("input_file", help="Path to CSV/Excel file with hackathon submissions")
    parser.add_argument("--output", "-o", default="manta_output", help="Output directory for analysis results")
    parser.add_argument("--github-token", "-g", default=os.environ.get("GITHUB_TOKEN"), help="GitHub API token")
    parser.add_argument("--anthropic-key", "-a", default=os.environ.get("ANTHROPIC_API_KEY"), help="Anthropic API key")
    parser.add_argument("--max-repos", "-m", type=int, default=None, help="Maximum number of repositories to process")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Check if API keys are provided
    if not args.github_token:
        raise ValueError("GitHub API token is required. Provide it with --github-token or set GITHUB_TOKEN environment variable.")
    if not args.anthropic_key:
        raise ValueError("Anthropic API key is required. Provide it with --anthropic-key or set ANTHROPIC_API_KEY environment variable.")
    
    try:
        # Initialize Manta Insights
        analyzer = MantaInsights(
            github_token=args.github_token,
            anthropic_api_key=args.anthropic_key,
            verbose=args.verbose
        )
        
        # Run analysis
        analyzer.analyze_submissions(
            input_file=args.input_file,
            output_dir=args.output,
            max_repos=args.max_repos
        )
        
        print(f"Analysis complete! Results saved to {args.output}")
        print(f"- Project reports: {os.path.join(args.output, 'reports')}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())

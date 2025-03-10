# Manta Insights

## AI-Powered Hackathon Analysis System

Manta Insights is a streamlined tool designed to help hackathon organizers and judges evaluate submissions with the power of AI. The system leverages GitHub data and Claude's AI capabilities to provide comprehensive project analysis, categorization, and scoring.



![Adobe Express - file](https://github.com/user-attachments/assets/2cd0c6cd-795d-423f-aca6-177319c5b77c)


## Features

- **GitHub Repository Analysis**: Automatically extracts technologies used and README content
- **AI-Powered Evaluation**: Uses LLM's to analyze projects and generate insights
- **Project Categorization**: Classifies projects into types (Gaming, DeFi, Infrastructure, etc.)
- **Comprehensive Scoring**: Evaluates innovation, technical complexity, and presentation quality
- **Markdown Reports**: Generates detailed reports for each project with executive summaries

## Requirements

- Python 3.8+
- GitHub API token
- Anthropic API key (for Claude)
- Required Python packages: pandas, numpy, requests, PyGithub, anthropic, python-dotenv

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/manta-insights.git
cd manta-insights

# Install dependencies
pip install pandas numpy requests PyGithub anthropic python-dotenv
```

## Configuration

Create a `.env` file in the project directory with your API keys:

```
GITHUB_TOKEN=your_github_token_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Usage

### Basic Command

```bash
python manta_insights.py submissions.csv
```

### Full Options

```bash
python manta_insights.py submissions.csv \
  --output my_hackathon_results \
  --github-token YOUR_GITHUB_TOKEN \
  --anthropic-key YOUR_ANTHROPIC_KEY \
  --max-repos 50 \
  --verbose
```

### Input Format

Your CSV/Excel file should have these columns:
- `team_name`: Name of the team
- `project_name`: Name of the project
- `description`: Description of the project
- `github_url`: URL to GitHub repository

Example CSV:
```csv
team_name,project_name,description,github_url
"Code Wizards","Smart Calendar","An AI-powered calendar app that predicts tasks.","https://github.com/codewizards/smart-calendar"
```

## Output

The system generates:

1. **Project Reports**: Individual markdown files for each project containing:
   - Project details and classification
   - AI analysis of the project
   - Scores across multiple dimensions
   - Executive summary for judges

2. **Index File**: A summary document linking to all project reports

## Project Categories

Manta Insights classifies projects into these categories:
- **Gaming**: Games, game platforms, gaming-related tools
- **DeFi**: Decentralized finance, trading, financial applications
- **Infra**: Infrastructure, developer tools, frameworks
- **Entertainment**: Non-game entertainment, social, content creation
- **Security**: Security tools, auditing, protection
- **Other**: Projects that don't fit neatly in other categories

## License

MIT

## Acknowledgements
- Built with [PyGithub](https://github.com/PyGithub/PyGithub) for GitHub API integration

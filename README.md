# Ai_fakenews_detector

This is an AI-powered fake news and claim verification app that checks suspicious statements against live web evidence and returns an evidence-based verdict, confidence score, explanation, red flags, and source list.

It is designed for students, researchers, journalists, and everyday users who want a fast way to evaluate whether a claim looks likely true, likely false, misleading, mixed, or unverified.

---

## Live Demo

**Deployed App:** [Ai_fakenews_detector](https://huggingface.co/spaces/AhmadFiazAhmad/Ai_fakenews_detector)

---

## What Problem Does It Solve?

False information spreads quickly on social media, messaging apps, and news feeds. Many people read a claim and have no quick way to verify whether it is trustworthy.

This AI-Powered App solves this by:

- taking a claim as input,
- searching the web for live evidence,
- ranking sources by trust signals,
- analyzing the evidence with an AI model,
- and presenting a clear verdict with explanation.

This helps users make better decisions before believing or sharing information.

---

## Features

- **Claim verification** from pasted text, rumor, headline, or article snippet
- **Optional URL analysis** for related articles
- **Live web search** using Tavily to gather current evidence
- **Evidence ranking** using trust heuristics
- **AI-generated verdicts** such as:
  - Likely True
  - Likely False
  - Misleading
  - Unverified
  - Mixed Evidence
- **Confidence score** shown as a percentage
- **Detailed explanation** generated from evidence
- **Key points and red flags** for quick review
- **Source cards** with title, domain, trust label, and snippet
- **History panel** for previous verifications
- **Downloadable reports** in JSON and Markdown
- **Premium Streamlit UI** with custom styling and charts
- **Debug and settings controls** for tuning the experience

---

## AI Feature

AI uses a two-model LLM setup through Groq:

- **Primary model:** `openai/gpt-oss-120b`
- **Fallback model:** `llama-3.3-70b-versatile`

### How the AI works

1. The app first collects live evidence from Tavily.
2. The evidence is formatted into a structured context block.
3. A strict system prompt instructs the model to:
   - use only the provided evidence,
   - avoid inventing facts,
   - return valid JSON only,
   - provide a verdict, confidence, summary, explanation, key points, red flags, evidence used, recommendation, and safety note.

### Full System Prompt

```
You are a meticulous fact-checking AI for a premium content verification tool.

Your job:
- Use ONLY the evidence provided in the prompt.
- Do NOT invent facts.
- Do NOT browse the web by yourself.
- Compare conflicting sources carefully.
- Be precise, calm, and evidence-driven.

Return valid JSON only with this schema:
{
  "verdict": "Likely True | Likely False | Misleading | Unverified | Mixed Evidence",
  "confidence": 0-100,
  "short_summary": "one concise sentence",
  "explanation": "detailed explanation with evidence-based reasoning",
  "key_points": ["bullet 1", "bullet 2", "bullet 3"],
  "red_flags": ["warning 1", "warning 2"],
  "evidence_used": [
    {
      "title": "source title",
      "url": "source url",
      "snippet": "relevant excerpt",
      "trust": "High Trust | Medium Trust | Low Trust"
    }
  ],
  "recommendation": "what the user should do next",
  "safety_note": "gentle note when needed"
}

Writing rules:
- Keep the tone non-judgmental.
- If the claim is health-related, be extra careful and avoid medical advice.
- If the evidence is mixed, say so.
- If evidence is insufficient, say Unverified.
- JSON must be parseable by json.loads().
```

---

## Tools, Services, and Models Used

### Development
- Python
- Streamlit
- Requests
- BeautifulSoup4
- Plotly

### AI and Search
- Groq API
- Tavily API
  ## Environment Variables

Create a `.env` file or configure the following secrets:

```env
GROQ_API_KEY=your_key
TAVILY_API_KEY=your_key
```

### Models
- `openai/gpt-oss-120b`
- `llama-3.3-70b-versatile`

### Deployment
- Hugging Face Spaces

---

## How It Works

1. User enters a claim.
2. The app searches the web for supporting or conflicting evidence.
3. Sources are ranked using trust heuristics.
4. The AI model reads only the collected evidence.
5. The app returns:
   - verdict
   - confidence
   - explanation
   - key points
   - red flags
   - sources used

---

## Screenshots

Screenshots of Project, for example:

### 1. Main Interface
<img width="1920" height="1080" alt="Main Screen" src="https://github.com/user-attachments/assets/a7794a93-3fee-482f-945b-22a8f907707d" />



### 2. Claim Verification Result
<img width="1920" height="1080" alt="Reuslt" src="https://github.com/user-attachments/assets/b8f38ed1-55f9-4fa0-9afe-abf4ebaf1a0a" />


### 3. Evidence and Sources
<img width="1920" height="1080" alt="Evidences" src="https://github.com/user-attachments/assets/389fb67d-a296-4d35-8334-58762dbbddf6" />

<img width="1920" height="1080" alt="Analysis" src="https://github.com/user-attachments/assets/83255f86-0c7a-48e5-92d3-b966f8cb6101" />



---
## How to Run the Project

### Prerequisites
- Python 3.9+
- A [Groq API key](https://console.groq.com)
- A [Tavily API key](https://tavily.com)

### 1. Clone the repository
```bash
git clone https://github.com/ahmadfiazahmad/Ai_fakenews_detector.git
cd Ai_fakenews_detector
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your API keys
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```
App will open automatically at `http://localhost:8501`.

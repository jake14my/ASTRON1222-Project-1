# Eclipse Explorer

A comprehensive solar eclipse viewing advisor and visualization tool that helps you find and plan for solar eclipses from 2001-2100. By Marshal Houser and Jake Meyer.


## Features

- **Eclipse Chatbot**: LLM-powered advisor that answers questions about eclipses and suggests viewing locations
- **Eclipse Viewer**: Interactive visualization showing what an eclipse looks like from any location on Earth
- **Eclipse Data**: Scrapes and organizes NASA eclipse catalog data
- **Streamlit App**: Combined chatbot and visualization in a web interface

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key:
   - Create a `.env` file in the project root
   - Add your API key:
     ```
     ASTRO1221_API_KEY=your_api_key_here
     ```
     Note: This API key can be found on Carmen in the provided .env file

## Usage

### Streamlit App (Recommended)
Run the combined chatbot and visualization:
```bash
streamlit run eclipse_app.py
```

### Jupyter Notebooks

1. **EclipseData.ipynb**: Scrapes NASA eclipse data and exports to `eclipse_data.json`
2. **EclipseChatbot.ipynb**: Interactive chatbot for eclipse queries
3. **PathwayVisualization.ipynb**: Interactive eclipse viewer with time slider

## Project Structure

- `eclipse_app.py` - Streamlit web application
- `EclipseChatbot.ipynb` - Jupyter notebook chatbot
- `PathwayVisualization.ipynb` - Jupyter notebook visualization
- `EclipseData.ipynb` - Data collection notebook
- `eclipse_data.json` - Eclipse database (224 eclipses, 2001-2100)
- `eclipse_utils.py` - Shared utility functions and Eclipse class

## Example Queries

- "When is the next eclipse visible from Austin?"
- "Tell me about the 2026 total eclipse"
- "Where should I go to see the 2027 Aug total eclipse?"
- "What eclipses can I see from Tokyo in the next 20 years?"

## Data Source

Eclipse data is sourced from NASA's Solar Eclipse Catalog:
https://eclipse.gsfc.nasa.gov/SEcat5/SE2001-2100.html

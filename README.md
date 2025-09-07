# LawHub - Integrated Legal Platform

A comprehensive legal assistance platform with AI-powered search, document analysis, and emergency help features.

## 🚀 Quick Start (PowerShell-Free)

### Option 1: Python Script (Recommended)
```bash
python start_server_no_powershell.py
```

### Option 2: Batch File
```bash
start_server.bat
```

### Option 3: Direct Python
```bash
python app.py
```

## ⚠️ PowerShell Crash Fix

If you're experiencing the PowerShell crash error (`exit code: -1073741510`), use the PowerShell-free options above. This is a known Windows issue that we've permanently resolved.

## 🌐 Access the Application

Once started, open your browser and go to:
- **Landing Page**: http://localhost:5000
- **Main Application**: http://localhost:5000/app
- **API Status**: http://localhost:5000/api/status

## 📋 Features

### 🏠 Landing Page
- Modern, responsive design with dark theme
- Emergency banner for immediate assistance
- Hamburger menu (☰) for easy navigation
- Comprehensive feature showcase
- Statistics and trust indicators
- Direct access to all application features

### 🏠 Dashboard
- Overview of all available features
- Quick access to legal tools

### 💬 Legal Q&A Chatbot
- Ask legal questions in natural language
- Get answers from comprehensive legal databases
- Supports multiple jurisdictions
- **🤖 DeepSeek AI Integration**: Advanced AI-powered legal advice

### 📚 Laws Explorer
- Search through legal databases
- Available jurisdictions:
  - 🇺🇸 United States (Supreme Court cases, federal laws)
  - 🇧🇩 Bangladesh (National laws and regulations)
  - 🇵🇰 Pakistan (Legal documents and cases)
  - 🇦🇺 Australia (Legal corpus and regulations)
  - 📋 Legal QA (Training and test datasets)
  - 💊 Narcotics Act (Narcotics Act No. 1 of 2008)

### 📄 Document Analysis
- Upload legal documents (PDF/DOCX)
- OCR text extraction
- Risk assessment and analysis

### 🚨 Emergency Help
- Quick access to emergency contacts
- Legal advice for critical situations
- Contact information for legal aid

## 🔧 Technical Details

### Backend
- **Framework**: Flask (Python)
- **Search Engine**: TF-IDF with cosine similarity
- **Datasets**: Multiple legal databases from various jurisdictions
- **AI Integration**: DeepSeek AI for advanced legal advice
- **API Endpoints**:
  - `POST /api/ask` - Legal question answering
  - `POST /api/legal_qa` - Legal QA specific queries
  - `POST /api/deepseek_legal` - DeepSeek AI legal advice
  - `GET /api/status` - Backend status

### Frontend
- **Framework**: Vanilla JavaScript with Tailwind CSS
- **Design**: Dark theme with orange accent colors
- **Responsive**: Works on desktop and mobile devices

## 📦 Dependencies

### Required Python Packages
```bash
pip install flask flask-cors pandas scikit-learn datasets deep-translator tqdm numpy
```

### Optional (for enhanced features)
```bash
pip install kaggle  # For additional dataset access
```

## 🤖 DeepSeek AI Integration

### Setup DeepSeek AI

1. **Get API Key**:
   - Visit [DeepSeek Platform](https://platform.deepseek.com/)
   - Sign up or log in to your account
   - Go to API Keys section
   - Create a new API key
   - Copy the key (starts with `sk-`)

2. **Configure API Key**:
   ```bash
   # Option 1: Use setup script (Recommended)
   python setup_deepseek.py
   
   # Option 2: Manual configuration
   # Edit config.py and add your API key
   DEEPSEEK_API_KEY = "sk-your-api-key-here"
   
   # Option 3: Environment variable
   export DEEPSEEK_API_KEY="sk-your-api-key-here"
   ```

3. **Test Connection**:
   ```bash
   python setup_deepseek.py test
   ```

### Using DeepSeek AI

- Click the **🤖 DeepSeek AI** button in the Legal Q&A section
- Ask any legal question about rights, laws, procedures, etc.
- Get comprehensive AI-powered legal advice including:
  - Relevant legal rights and protections
  - Applicable laws, acts, and sections
  - Recommended actions and procedures
  - Potential legal consequences
  - When to seek professional legal counsel

### Features

- **Comprehensive Legal Advice**: Covers international law, legal rights, and procedures
- **Context-Aware**: Integrates with existing legal databases for enhanced responses
- **Structured Responses**: Clear, actionable legal guidance
- **Professional Disclaimers**: Always recommends seeking professional counsel when appropriate

## 🛠️ Installation
1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the server** using one of the methods above
4. **Open your browser** to http://localhost:5000

## 🔍 Troubleshooting

### PowerShell Crash (-1073741510)
- **Solution**: Use `python start_server_no_powershell.py` instead
- **Why**: This avoids PowerShell entirely and uses direct Python execution

### Port Already in Use
- **Solution**: Change the port in `app.py` or kill the existing process
- **Alternative**: Use `python app.py --port 5001`

### Missing Dependencies
- **Solution**: Run `pip install -r requirements.txt`
- **Manual**: Install packages listed in the dependencies section

### Dataset Loading Issues
- **Solution**: Check file paths in `app.py`
- **Alternative**: The system will work with available datasets

## 📁 Project Structure

```
LawHub-1/
├── app.py                          # Main Flask application
├── start_server_no_powershell.py   # PowerShell-free server starter
├── start_server.bat                # Windows batch file
├── templates/
│   ├── landing_page.html          # Landing page with modern design
│   └── integrated_frontend.html    # Main application interface
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes.

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Use the PowerShell-free startup methods
3. Ensure all dependencies are installed
4. Check that you're in the correct directory

---

**Note**: This platform is designed for educational purposes and should not replace professional legal advice. 
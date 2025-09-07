# LawHub Setup Instructions

## Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project_LawHub
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   ```bash
   # Copy the template config file
   copy config_template.py config.py
   
   # Edit config.py and add your DeepSeek API key
   # DEEPSEEK_API_KEY = "sk-your-api-key-here"
   ```

4. **Start the application**
   ```bash
   # Option 1: Use the PowerShell-free starter (Recommended)
   python start_server_no_powershell.py
   
   # Option 2: Use batch file
   start_server.bat
   
   # Option 3: Direct Python
   python app.py
   ```

5. **Access the application**
   - Landing Page: http://localhost:5000
   - Main Application: http://localhost:5000/app

## DeepSeek AI Setup

1. **Get API Key**:
   - Visit [DeepSeek Platform](https://platform.deepseek.com/)
   - Sign up or log in to your account
   - Go to API Keys section
   - Create a new API key
   - Copy the key (starts with `sk-`)

2. **Configure API Key**:
   - Copy `config_template.py` to `config.py`
   - Edit `config.py` and add your API key:
     ```python
     DEEPSEEK_API_KEY = "sk-your-api-key-here"
     ```

3. **Test Connection**:
   ```bash
   python setup_deepseek.py test
   ```

## Troubleshooting

### PowerShell Crash (-1073741510)
- **Solution**: Use `python start_server_no_powershell.py` instead
- **Why**: This avoids PowerShell entirely and uses direct Python execution

### Port Already in Use
- **Solution**: Change the port in `config.py` or kill the existing process
- **Alternative**: Use `python app.py --port 5001`

### Missing Dependencies
- **Solution**: Run `pip install -r requirements.txt`

### Dataset Loading Issues
- **Solution**: The system will work with available datasets and can regenerate cache files

## Important Notes

- The `config.py` file is excluded from git for security (contains API keys)
- Cache files (*.pkl) are excluded from git as they are large and can be regenerated
- The `data/` directory is excluded from git due to large dataset files
- Kaggle credentials are excluded from git for security

## File Structure

```
Project_LawHub/
├── app.py                          # Main Flask application
├── config_template.py              # Configuration template
├── start_server_no_powershell.py   # PowerShell-free server starter
├── start_server.bat                # Windows batch file
├── templates/
│   ├── landing_page.html          # Landing page
│   └── integrated_frontend.html    # Main application interface
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── SETUP.md                        # This setup guide

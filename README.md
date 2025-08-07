# 📊 Plotly Navigator 🚀  
*Your data. Your vibe. Your plot.*

Welcome to **Plotly Navigator** — a zero-code, all-fun data viz playground for analysts, students, and anyone who loves turning spreadsheets into stories.

Try it online 👉 *https://plotlynavigator.streamlit.app/*

---

## 🎯 What It Does  
Upload your CSV or Excel file, and *BOOM!*  
Instantly explore:

- 🔍 Distributions (Histograms, Box Plots, Violin, ECDFs)
- 🧩 Relationships (Scatter, Bar, Violin, Box)
- 🧠 Time trends (line charts with smart date handling)
- 🌈 Category breakdowns (sweet sweet pie charts)
- 🪐 Multi-dimensional madness (correlation heatmaps, bubble charts, pair plots)

It’s like Tableau, but Streamlit-powered, Plotly-stylized, and snack-sized.

---

## 🧠 Who Is It For?  

- You love clicking more than coding  
- You have data, but not the time to build charts from scratch  
- You appreciate a tooltip that says “Layman’s Interpretation.”

---

## 🚀 Installation

### Option 1: One-Click macOS Installer

```bash
bash installer-macos-universal.sh
```

This will:
- Detect your Mac architecture (Intel or Apple Silicon)
- Install Miniforge (if needed)
- Create the `plotlynavigator` environment
- Add a Desktop shortcut using Automator【77†source】

---

### Option 2: One-Click Windows Installer

```powershell
Right-click → Run with PowerShell → installer-windows.ps1
```

This will:
- Detect your Anaconda or Miniconda installation
- Create/update the `plotlynavigator` conda environment using `__environment__.yml`
- Create a desktop shortcut (`Start Plotly Navigator App`)
- Generate a launcher and uninstaller script

> 💡 **Note**: Make sure Conda is installed before running.

---

### Option 3: Manual Setup

```bash
git clone https://github.com/teonghan/plotlynavigator.git
cd plotly-navigator
pip install -r requirements.txt
streamlit run app.py
```

Then just upload a `.csv` or `.xlsx` and go wild.

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io) for the snappy UI  
- [Plotly](https://plotly.com/python/) for visual sugar  
- [Pandas](https://pandas.pydata.org) & [NumPy](https://numpy.org) because… duh【79†source】

---

## 🤓 Bonus Perks  

- Built-in chart explanations (no more Googling “what is a violin plot?”)  
- Auto-detection of column types  
- Snippets of Plotly code in case you wanna take it further

---

## 📃 License

MIT License — free to use, remix, and have fun!

---

Unleash the power of your spreadsheet — one click at a time.

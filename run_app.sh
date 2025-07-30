#!/bin/bash
source $HOME/miniforge3/bin/activate plotlynavigator
cd "/Users/teonghan/Sync/App/plotlynavigator"
streamlit run app.py
read -p "Press ENTER to exit the app..."

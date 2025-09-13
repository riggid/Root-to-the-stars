#!/usr/bin/env python3
"""
main.py - Main application entry point for Space Trajectory Planner
Starts the FastAPI backend server.
"""

import uvicorn
import os

if __name__ == "__main__":
    print("🚀 Starting Space Mission Planner backend...")
    print("✅ Frontend available at: http://127.0.0.1:8000")
    print("📚 API docs available at: http://127.0.0.1:8000/docs")
    os.system('fuser -k 8000/tcp')
    uvicorn.run("src.backend:app", host="127.0.0.1", port=8000, reload=True)
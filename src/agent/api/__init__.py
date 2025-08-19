# src/stock_assistant/api/__init__.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import agent, health
from ..shared.settings.settings import settings

def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Vietnamese Stock Market ReAct Agent API",
        debug=settings.app.debug
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(agent.router)
    app.include_router(health.router)
    
    @app.get("/")
    async def root():
        return {
            "message": "Welcome to Milano-Stock Agent",
            "version": settings.app.version,
            "docs": "/docs"
        }
    
    return app
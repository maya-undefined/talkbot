import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

DB_URL = os.getenv("DATABASE_URL")  # e.g., postgres+psycopg://user:pass@host:5432/db

engine = create_engine(DB_URL, pool_pre_ping=True) if DB_URL else None
SessionLocal = sessionmaker(bind=engine) if engine else None
Base = declarative_base()
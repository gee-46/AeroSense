from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings
from app.core.logging import logger

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(settings.DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_missing_columns(bind=None) -> None:
    """Minimal forward-only migration.

    `create_all` never alters an existing table, so a database created by an
    earlier version of the app would lack columns added since. Add any that are
    missing (nullable, so old rows stay valid) instead of failing at query time.
    """
    bind = bind or engine
    inspector = inspect(bind)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            if not column.nullable and column.default is None and column.server_default is None:
                logger.error(f"Cannot auto-migrate NOT NULL column {table.name}.{column.name}")
                continue
            col_type = column.type.compile(dialect=bind.dialect)
            with bind.begin() as conn:
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'))
            logger.info(f"Migrated: added column {table.name}.{column.name}")


def init_db():
    from app.db import models  # noqa: F401  (ensure models are registered)
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def dispose_db() -> None:
    engine.dispose()

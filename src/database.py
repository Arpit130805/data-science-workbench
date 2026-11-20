import json
from datetime import datetime
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./workbench.db",
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)
    row_count = Column(Integer, nullable=False)
    column_count = Column(Integer, nullable=False)
    missing_value_count = Column(Integer, nullable=False)
    duplicate_row_count = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    rows = relationship(
        "DatasetRow",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class DatasetRow(Base):
    __tablename__ = "dataset_rows"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(
        Integer,
        ForeignKey("datasets.id"),
        nullable=False,
    )
    row_number = Column(Integer, nullable=False)
    row_data = Column(JSON, nullable=False)

    dataset = relationship("Dataset", back_populates="rows")


def create_tables():
    """Create database tables if they do not already exist."""
    Base.metadata.create_all(bind=engine)


def save_dataset(file_name, dataframe):
    """Save dataset metadata and every row into PostgreSQL."""
    create_tables()

    extension = file_name.rsplit(".", maxsplit=1)[-1].lower()

    dataset = Dataset(
        name=file_name,
        file_type=extension,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        missing_value_count=int(dataframe.isna().sum().sum()),
        duplicate_row_count=int(dataframe.duplicated().sum()),
    )

    safe_records = json.loads(
        dataframe.to_json(
            orient="records",
            date_format="iso",
        )
    )

    session = SessionLocal()

    try:
        session.add(dataset)
        session.flush()

        for row_number, row_data in enumerate(safe_records, start=1):
            dataset_row = DatasetRow(
                dataset_id=dataset.id,
                row_number=row_number,
                row_data=row_data,
            )
            session.add(dataset_row)

        session.commit()

        return dataset.id

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


def get_saved_datasets():
    """Return metadata for every saved dataset."""
    create_tables()

    session = SessionLocal()

    try:
        datasets = (
            session.query(Dataset)
            .order_by(Dataset.uploaded_at.desc())
            .all()
        )

        return [
            {
                "ID": dataset.id,
                "Dataset Name": dataset.name,
                "File Type": dataset.file_type,
                "Rows": dataset.row_count,
                "Columns": dataset.column_count,
                "Missing Values": dataset.missing_value_count,
                "Duplicate Rows": dataset.duplicate_row_count,
                "Uploaded At": dataset.uploaded_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            for dataset in datasets
        ]

    finally:
        session.close()
        
def run_select_query(query):
    """Run one safe read-only SQL query and return its results."""

    clean_query = query.strip().rstrip(";")
    normalized_query = clean_query.lower()

    blocked_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "grant",
        "revoke",
    ]

    if not normalized_query.startswith(("select", "with")):
        raise ValueError("Only SELECT queries are allowed.")

    if ";" in clean_query:
        raise ValueError("Only one SQL query can be executed at a time.")

    if any(keyword in normalized_query for keyword in blocked_keywords):
        raise ValueError("Only safe read-only SELECT queries are allowed.")

    with engine.connect() as connection:
        result = connection.execute(text(clean_query))

        return pd.DataFrame(
            result.fetchall(),
            columns=result.keys(),
        )
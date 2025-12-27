from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from datetime import datetime
import time

engine = create_engine("sqlite:///fileStore.db", echo=False)
Base = declarative_base()

class Files(Base):
    __tablename__ = "files"
    file_id : Mapped[int] = mapped_column(primary_key=True)
    faiss_index: Mapped[int] = mapped_column(unique=True, nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(500), index=True, unique=True)
    file_name: Mapped[str] = mapped_column(String(100), index=True)
    file_hash: Mapped[str] = mapped_column(String(64), index=True)
    modified_time: Mapped[datetime]


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_file(file_path: str, file_name: str, file_hash: str, modified_time: str, faiss_index: int) -> Files:
    db = SessionLocal()
    try:
        modified_dt = datetime.fromtimestamp(time.mktime(time.strptime(modified_time)))
        file_record = Files(
            faiss_index=faiss_index,
            file_path=file_path,
            file_name=file_name,
            file_hash=file_hash,
            modified_time=modified_dt
        )
        db.add(file_record)
        db.commit()
        db.refresh(file_record)
        return file_record
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def create_files_bulk(files_data: List[tuple[str, str, str, str, int]]) -> int:
    """Bulk insert multiple files (much faster than individual inserts)

    Args:
        files_data: List of tuples (file_path, file_name, file_hash, modified_time, faiss_index)

    Returns:
        Number of files inserted
    """
    db = SessionLocal()
    try:
        file_records = []
        for file_path, file_name, file_hash, modified_time, faiss_index in files_data:
            modified_dt = datetime.fromtimestamp(time.mktime(time.strptime(modified_time)))
            file_records.append(Files(
                faiss_index=faiss_index,
                file_path=file_path,
                file_name=file_name,
                file_hash=file_hash,
                modified_time=modified_dt
            ))

        db.bulk_save_objects(file_records)
        db.commit()
        return len(file_records)
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_file_by_id(file_id: int) -> Optional[Files]:
    db = SessionLocal()
    try:
        return db.query(Files).filter(Files.file_id == file_id).first()
    finally:
        db.close()

def get_file_by_path(file_path: str) -> Optional[Files]:
    db = SessionLocal()
    try:
        return db.query(Files).filter(Files.file_path == file_path).first()
    finally:
        db.close()

def get_file_by_hash(file_hash: str) -> Optional[Files]:
    db = SessionLocal()
    try:
        return db.query(Files).filter(Files.file_hash == file_hash).first()
    finally:
        db.close()

def get_files_by_name(file_name: str) -> List[Files]:
    db = SessionLocal()
    try:
        return db.query(Files).filter(Files.file_name == file_name).all()
    finally:
        db.close()

def get_all_files() -> List[Files]:
    db = SessionLocal()
    try:
        return db.query(Files).all()
    finally:
        db.close()

def get_file_by_faiss_index(faiss_index: int) -> Optional[Files]:
    db = SessionLocal()
    try:
        return db.query(Files).filter(Files.faiss_index == faiss_index).first()
    finally:
        db.close()

def get_files_by_faiss_indices(faiss_indices: List[int]) -> List[Files]:
    db = SessionLocal()
    try:
        return db.query(Files).filter(Files.faiss_index.in_(faiss_indices)).all()
    finally:
        db.close()

def update_file(file_id: int, file_path: Optional[str] = None, file_name: Optional[str] = None,
                file_hash: Optional[str] = None, modified_time: Optional[str] = None,
                faiss_index: Optional[int] = None) -> Optional[Files]:
    db = SessionLocal()
    try:
        file_record = db.query(Files).filter(Files.file_id == file_id).first()
        if not file_record:
            return None

        if file_path is not None:
            file_record.file_path = file_path
        if file_name is not None:
            file_record.file_name = file_name
        if file_hash is not None:
            file_record.file_hash = file_hash
        if modified_time is not None:
            modified_dt = datetime.fromtimestamp(time.mktime(time.strptime(modified_time)))
            file_record.modified_time = modified_dt
        if faiss_index is not None:
            file_record.faiss_index = faiss_index

        db.commit()
        db.refresh(file_record)
        return file_record
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def delete_file(file_id: int) -> Optional[int]:
    """Delete a file and return its FAISS index for removal from FAISS index"""
    db = SessionLocal()
    try:
        file_record = db.query(Files).filter(Files.file_id == file_id).first()
        if not file_record:
            return None
        faiss_idx = file_record.faiss_index
        db.delete(file_record)
        db.commit()
        return faiss_idx
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def delete_file_by_path(file_path: str) -> Optional[int]:
    """Delete a file by path and return its FAISS index for removal from FAISS index"""
    db = SessionLocal()
    try:
        file_record = db.query(Files).filter(Files.file_path == file_path).first()
        if not file_record:
            return None
        faiss_idx = file_record.faiss_index
        db.delete(file_record)
        db.commit()
        return faiss_idx
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def delete_all_files():
    """Delete all files from the database"""
    db = SessionLocal()
    try:
        db.query(Files).delete()
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
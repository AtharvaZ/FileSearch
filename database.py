from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from datetime import datetime
import time

engine = create_engine("sqlite:///fileStore.db", echo=True)
Base = declarative_base()

class Files(Base):
    __tablename__ = "files"
    file_id : Mapped[int] = mapped_column(primary_key=True)
    file_path: Mapped[str] = mapped_column(String(500), index=True, unique=True)
    file_name: Mapped[str] = mapped_column(String(100), index=True)
    file_hash: Mapped[str] = mapped_column(String(64), index=True)
    modified_time: Mapped[datetime]


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_file(file_path: str, file_name: str, file_hash: str, modified_time: str) -> Files:
    db = SessionLocal()
    try:
        modified_dt = datetime.fromtimestamp(time.mktime(time.strptime(modified_time)))
        file_record = Files(
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

def update_file(file_id: int, file_path: Optional[str] = None, file_name: Optional[str] = None, 
                file_hash: Optional[str] = None, modified_time: Optional[str] = None) -> Optional[Files]:
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
        
        db.commit()
        db.refresh(file_record)
        return file_record
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def delete_file(file_id: int) -> bool:
    db = SessionLocal()
    try:
        file_record = db.query(Files).filter(Files.file_id == file_id).first()
        if not file_record:
            return False
        db.delete(file_record)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def delete_file_by_path(file_path: str) -> bool:
    db = SessionLocal()
    try:
        file_record = db.query(Files).filter(Files.file_path == file_path).first()
        if not file_record:
            return False
        db.delete(file_record)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from agent_dbmanager.db.models import Base, Supplier, SupplierLocation
from logger import logger

DATABASE_URL = "postgresql+psycopg2://admin:admin@localhost:5432/suppliersearchagentdb"

engine = create_engine(
    DATABASE_URL,
    echo=False
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def init_db():
    Base.metadata.create_all(bind=engine)

def save_supplier_to_db(data: dict):
    session = SessionLocal()
    try:
        supplier = Supplier(
            name=data.get("name", ""),
            website=data.get("website", ""),
            description=data.get("description", ""),
            email=data.get("email", []),
            phone=data.get("phone", []),
            vat_number=data.get("vat_number", "")
        )

        for loc in data.get("locations", []):
            location = SupplierLocation(
                country=loc.get("country", ""),
                city=loc.get("city", ""),
                address=loc.get("address", "")
            )
            supplier.locations.append(location)

        session.add(supplier)
        session.commit()

        logger.info(f"[AGENT-DBMANAGER] Supplier saved successfully: {supplier.name}")
        return supplier.id
    except Exception as e:
        session.rollback()
        logger.error(f"[AGENT-DBMANAGER] Error saving supplier: {e}")
        raise e
    finally:
        session.close()

def execute_search_query(location: str):
    """
    Generates and executes a raw SQL query in the database based on location.
    Returns an array of dictionaries (the extracted results).
    """
    session = SessionLocal()
    try:

        sql_query = """
            SELECT s.name, s.website, s.description, s.email, s.phone, s.vat_number,
                sl.id AS location_id, sl.country, sl.city, sl.address
            FROM supplier s
            LEFT JOIN supplier_locations sl ON s.id = sl.supplier_id
            WHERE 1=1
        """
        params = {}
            
        if location and location.strip():
            sql_query += " AND sl.city ILIKE :loc"
            params["loc"] = f"%{location.strip()}%"

        result = session.execute(text(sql_query), params)
        
        suppliers = [dict(row._mapping) for row in result]
        logger.info(f"[AGENT-DBMANAGER] Found {len(suppliers)} suppliers for location '{location}'")
        
        return suppliers
    except Exception as e:
        logger.error(f"[AGENT-DBMANAGER] Error executing SQL search query: {e}")
        raise e
    finally:
        session.close()